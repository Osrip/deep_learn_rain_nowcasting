import einops
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from helper.helper_functions import create_dilation_list


class LayerNormChannelOnly(nn.Module):
    """
    This LayerNorm only normalizes across the channel vector for each spatial pixel.
    Assumes input tensor to have shape b x c x h x w
    Implemented as in original ConvNeXt code:
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """
    def __init__(self, c: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.layer_norm(x)
        x = einops.rearrange(x, 'b h w c -> b c h w')
        return x


class SkipConnection(nn.Module):
    """
    Skip connection implementation by Manu
    avg. pooling for downscaling (both in space and channel), using einops.reduce
    Simply duplicating elemnts for upscaling (both space and channel), using einops.repeat
    Manu experienced better gradient pass-through compared to to conv rescaling in the scip connection.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            spatial_factor: float = 1.0
    ):
        super(SkipConnection, self).__init__()
        assert spatial_factor == 1 or int(spatial_factor) > 1 or int(
            1 / spatial_factor) > 1, f'invalid spatial scale factor in SpikeFunction: {spatial_factor}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_factor = spatial_factor

    def channel_skip(self, input: torch.Tensor):
        in_channels = self.in_channels
        out_channels = self.out_channels

        if in_channels == out_channels:
            return input

        if in_channels % out_channels == 0 or out_channels % in_channels == 0:

            if in_channels > out_channels:
                return einops.reduce(input, 'b (c n) h w -> b c h w', 'mean', n=in_channels // out_channels)

            if out_channels > in_channels:
                return einops.repeat(input, 'b c h w -> b (c n) h w', n=out_channels // in_channels)

        mean_channels = np.gcd(in_channels, out_channels)
        input = einops.reduce(input, 'b (c n) h w -> b c h w', 'mean', n=in_channels // mean_channels)
        return einops.repeat(input, 'b c h w -> b (c n) h w', n=out_channels // mean_channels)

    def scale_skip(self, input: torch.Tensor):
        spatial_factor = self.spatial_factor

        if spatial_factor == 1:
            return input

        if spatial_factor > 1:
            return einops.repeat(
                input,
                'b c h w -> b c (h h2) (w w2)',
                h2=int(spatial_factor),
                w2=int(spatial_factor)
            )

        height = input.shape[2]
        width = input.shape[3]

        # scale factor < 1
        spatial_factor = int(1 / spatial_factor)

        if width % spatial_factor == 0 and height % spatial_factor == 0:
            return einops.reduce(
                input,
                'b c (h h2) (w w2) -> b c h w',
                'mean',
                h2=spatial_factor,
                w2=spatial_factor
            )

        if width >= spatial_factor and height >= spatial_factor:
            return nn.functional.avg_pool2d(
                input,
                kernel_size=spatial_factor,
                stride=spatial_factor
            )

        assert width > 1 or height > 1
        return einops.reduce(input, 'b c h w -> b c 1 1', 'mean')

    def forward(self, input: torch.Tensor):

        if self.spatial_factor > 1:
            return self.scale_skip(self.channel_skip(input))

        return self.channel_skip(self.scale_skip(input))


class ConvNext(nn.Module):
    """
    ConvNext Block
    c: num of channels
    """
    def __init__(self, c: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(c, c*4, kernel_size=7, groups=c, padding='same')  # groups=c makes this a depth wise convolution
        self.layer_norm = LayerNormChannelOnly(c*4)
        self.conv_2 = nn.Conv2d(c*4, c, kernel_size=1)
        self.conv_3 = nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x: torch.Tensor):
        skip = x
        x = self.conv_1(x)
        x = self.layer_norm(x)
        x = self.conv_2(x)
        x = F.leaky_relu(x)
        x = self.conv_3(x)
        x += skip
        return x


class ConvNextDownScale(nn.Module):
    """
    Downscaling ConvNext block
    """
    def __init__(self, c: int, c_factor: int, spatial_factor: int):
        super().__init__()
        #Does the conv 2 2 with stride 2 cause image patching? If so try conv 3x3 with padding
        self.conv = nn.Conv2d(c, int(c * c_factor), kernel_size=spatial_factor, stride=spatial_factor)
        self.layer_norm = LayerNormChannelOnly(int(c * c_factor))
        self.skip = nn.Conv2d(c, int(c * c_factor), kernel_size=1, stride=spatial_factor)
        self.skip = SkipConnection(c, int(c * c_factor), 1/spatial_factor)

    def forward(self, x: torch.Tensor):
        skip = x
        skip = self.skip(skip)

        x = self.conv(x)
        x = self.layer_norm(x)

        x += skip
        return x


class ConvNextUpScale(nn.Module):
    """
    Upscaling ConvNext block
    """

    def __init__(self, c: int, c_factor: int, spatial_factor: int):
        super().__init__()
        if c % c_factor != 0:
            raise ValueError('Channel number has to divisible by scale_factor.')
        # Does the conv 2 2 with stride 2 cause image patching? If so try conv 3x3 with padding
        self.conv_transposed = nn.ConvTranspose2d(c, c//c_factor, kernel_size=spatial_factor, stride=spatial_factor,)
        self.layer_norm = LayerNormChannelOnly(c//c_factor)
        self.skip = nn.ConvTranspose2d(c, c//c_factor, kernel_size=1, stride=spatial_factor)
        self.skip = SkipConnection(c, c//c_factor, spatial_factor)


    def forward(self, x: torch.Tensor):
        skip = x
        skip = self.skip(skip)

        x = self.conv_transposed(x)
        x = self.layer_norm(x)

        x += skip
        return x


class EncoderModule(nn.Module):
    """
    One block of the Encoder that includes two ConvNext Block and a downsampling block
    """
    # Potentially implement this with nn.Sequencial to make number of blocks per module as well as c_in and c_out variable
    def __init__(self, c: int, c_factor: int, spatial_factor: int):
        super().__init__()
        self.conv_next_1 = ConvNext(c)
        self.conv_next_2 = ConvNext(c)
        self.conv_next_down_scale = ConvNextDownScale(c, c_factor, spatial_factor)

    def forward(self, x: torch.Tensor):
        x = self.conv_next_1(x)
        x = self.conv_next_2(x)
        x = self.conv_next_down_scale(x)
        return x


class DecoderModule(nn.Module):
    """
    One block of the Decoder that includes two ConvNext Block and an upsampling block
    """
    def __init__(self, c: int, c_factor: int, spatial_factor: int):
        super().__init__()
        self.conv_next_1 = ConvNext(c)
        self.conv_next_2 = ConvNext(c)
        self.conv_next_up_scale = ConvNextUpScale(c, c_factor, spatial_factor)

    def forward(self, x: torch.Tensor):
        x = self.conv_next_1(x)
        x = self.conv_next_2(x)
        x = self.conv_next_up_scale(x)
        return x


class Encoder(nn.Module):
    """
    Encoder
    This returns the whole list for the skip connections
    """
    def __init__(self, c_in: int, num_downscales: int, c_factor: int, spatial_factor: int):
        super().__init__()

        self.module_list = nn.ModuleList()
        c_curr = c_in
        for i in range(num_downscales):
            # 256 / 2 ** 5 = 8
            i += 1
            self.module_list.add_module(
                name='encoder_module_{}'.format(i),
                module=EncoderModule(c_curr, c_factor, spatial_factor)
            )
            c_curr = int(c_curr * c_factor)

    def forward(self, x: torch.Tensor):
        skip_list = []
        skip_list.append(x)  # A skip connection right from the input to the output
        for module in self.module_list:
            x = module(x)
            skip_list.append(x)
        return skip_list


class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, c: int, num_upscales: int, c_factor: int, spatial_factor: int):
        super().__init__()

        self.module_list = nn.ModuleList()
        c_curr = c
        for i in range(num_upscales):
            # 256 / 2 ** 5 = 8
            i += 1
            self.module_list.add_module(
                name=r'decoder_module_{i}',
                module=DecoderModule(c_curr, c_factor, spatial_factor)
            )
            c_curr = c_curr // c_factor

    def forward(self, skip_list: list):
        x = skip_list[0]
        test_list = []
        for i, (module, skip) in enumerate(zip(self.module_list, reversed(skip_list[1:]))):
            x = module(x)
            test_list.append(x)
            # x += skip
        return x


class UNet(nn.Module):
    def __init__(self,
                 c_in: int,
                 num_scalings: int = 4,
                 c_factor: int = 2,
                 spatial_factor: int = 2):
        super().__init__()
        self.encoder = Encoder(c_in, num_scalings, c_factor, spatial_factor)
        # TODO: is num_scalings -1 correct?
        c_latent = c_in * c_factor ** (num_scalings - 1)
        self.decoder = Decoder(c_latent, num_scalings, c_factor, spatial_factor)

    def forward(self, x: torch.Tensor):
        skip_list = self.encoder(x)
        x = self.decoder(skip_list)
        return x





