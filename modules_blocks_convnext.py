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
        else:
            raise ValueError('in_channels % out_channels is not 0')


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
    num of input and output channels are equal
    """
    def __init__(self, c: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(c, c*4, kernel_size=7, groups=c, stride=1, padding='same')  # groups=c makes this a depth wise convolution
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
    First normalize, then downscale acc to paper.
    In original paper, they do not implement a skip connection here.
    I implemented Manu's skip connection with avg pool for downsampling / duplication for upsampling
    """
    def __init__(self, c_in: int, c_out: int, spatial_factor: int):
        super().__init__()
        #Does the conv 2 2 with stride 2 cause image patching? If so try conv 3x3 with padding
        self.layer_norm = LayerNormChannelOnly(c_in)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=spatial_factor, stride=spatial_factor)
        # self.skip = nn.Conv2d(c_in, c_out, kernel_size=1, stride=spatial_factor)
        self.skip = SkipConnection(c_in, c_out, 1/spatial_factor)

    def forward(self, x: torch.Tensor):
        skip = self.skip(x)

        x = self.layer_norm(x)
        x = self.conv(x)

        x += skip
        return x


class ConvNextUpScale(nn.Module):
    """
    Upscaling ConvNext block
    First normalize, then downscale acc to paper.
    In original paper, they do not implement a skip connection here.
    I implemented Manu's skip connection with avg pool for downsampling / duplication for upsampling
    """
    def __init__(self, c_in: int, c_out: int, spatial_factor: int):
        super().__init__()
        if c_in % c_out != 0:
            raise ValueError('Channel number must be divisible by c_out.')
        # Upscaling using transposed convolution
        self.layer_norm = LayerNormChannelOnly(c_in)
        self.conv_transposed = nn.ConvTranspose2d(c_in, c_out, kernel_size=spatial_factor, stride=spatial_factor)
        # Adding a skip connection
        self.skip = SkipConnection(c_in, c_out, spatial_factor)

    def forward(self, x: torch.Tensor):
        skip = self.skip(x)

        x = self.layer_norm(x)
        x = self.conv_transposed(x)

        x += skip
        return x


class EncoderModule(nn.Module):
    """
    One block of the Encoder that includes two ConvNext Block and a downsampling block
    num_blocks: number of ConvNext blocks in the module (0 means only downsampling)
    First down-scaling block is applied, then the given number of convnext blocks
    This way Skip connections are applied right before the down sampling after the latent space processing by the
    ConvNeXt blocks
    """
    def __init__(self, c_in: int, c_out: int, spatial_factor: int, num_blocks: int):
        super().__init__()
        self.conv_next_down_scale = ConvNextDownScale(c_in, c_out, spatial_factor)
        self.conv_next_blocks = nn.Sequential(*[ConvNext(c_out) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor):
        x = self.conv_next_down_scale(x)
        x = self.conv_next_blocks(x)
        return x


class DecoderModule(nn.Module):
    """
    One block of the Decoder that includes two ConvNext Block and an upsampling block.
    num_blocks: number of ConvNext blocks in the module (0 means only upsampling).
    First up-scaling block is applied, then the given number of convnext blocks
    """
    def __init__(self, c_in: int, c_out: int, spatial_factor: int, num_blocks: int):
        super().__init__()
        self.conv_next_up_scale = ConvNextUpScale(c_in, c_out, spatial_factor)
        self.conv_next_blocks = nn.Sequential(*[ConvNext(c_out) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor):
        x = self.conv_next_up_scale(x)
        x = self.conv_next_blocks(x)
        return x


class Encoder(nn.Module):
    """
    Encoder
    This returns the whole list for the skip connections
    """
    def __init__(self, c_list: list[int], spatial_factor_list: list[int], num_blocks_list: list[int]):
        super().__init__()

        self.module_list = nn.ModuleList()

        for i, (c_in, c_out, spatial_factor, num_blocks) \
                in enumerate(zip(c_list[:-1], c_list[1:], spatial_factor_list, num_blocks_list)):
            # 256 / 2 ** 5 = 8
            i += 1
            self.module_list.add_module(
                name='encoder_module_{}'.format(i),
                module=EncoderModule(c_in, c_out, spatial_factor, num_blocks)
            )

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
    This returns the whole list for the skip connections
    """
    def __init__(self, c_list: list[int], spatial_factor_list: list[int], num_blocks_list: list[int]):
        super().__init__()

        self.module_list = nn.ModuleList()

        reversed_c_list = list(reversed(c_list))
        reversed_spatial_factors = list(reversed(spatial_factor_list))
        reversed_num_blocks = list(reversed(num_blocks_list))

        for i, (c_in, c_out, spatial_factor, num_blocks) in enumerate(zip(reversed_c_list[:-1],
                                                                          reversed_c_list[1:],
                                                                          reversed_spatial_factors,
                                                                          reversed_num_blocks)):
            # 256 / 2 ** 5 = 8
            i += 1
            self.module_list.add_module(
                name='decoder_module_{}'.format(i),
                module=DecoderModule(c_in, c_out, spatial_factor, num_blocks)
            )

    def forward(self, skip_list: list):
        x = skip_list[-1]  # Start with the first element from the skip connections
        reversed_skips = reversed(skip_list[:-1])
        for module, skip in zip(self.module_list, reversed_skips):
            x = module(x)
            x += skip  # Add the skip connection to the output of the current module
        return x


class DownScaleToTarget(nn.Module):
    '''
    Simply enter cropping and downscaling channels
    '''
    def __init__(self, c_in, c_out, height_width_target):
        super().__init__()
        self.center_crop = T.CenterCrop(size=height_width_target)
        self.down_scale = ConvNextDownScale(c_in, c_out, spatial_factor=1)

    def forward(self, x: torch.Tensor):
        x = self.center_crop(x)
        x = self.down_scale(x)
        return x


class ConvNeXtUNet(nn.Module):
    """
    ConvNeXt Unet
    c_list: list that includes the number of channels that each scaling module should input / output
            The length of the list corresponds to the (number of downscalings = number of upscalings) + 1
            in the UNet. So the first list entry is the number of input channels
    spatial_factor_list: list that includes the spatial downscaling / upscaling factors for ech down / upscaling
            module. The length of the list corresponds to the number of downscalings = number of upscalings
    num_blocks_list: List of the number of ConvNeXt blocks for each downscaling / upscaling. The downsampling / upsampling
            block itself is not included in the number. The length of the list corresponds to the number of
            downscalings = number of upscalings
    Just like in the paper the skip connections are applied right before the down sampling
    As of now we also implement a skip connection right from input to output (unlike paper)

    Example usage:
                self.model = ConvNeXtUNet(
                c_list=[4, 32, 64, 128, 256],
                spatial_factor_list=[2, 2, 2, 2],
                num_blocks_list=[2, 2, 2, 2],
                c_target=s_num_bins_crossentropy,
                height_width_target=s_width_height_target
            )
    """
    # TODO: Initialize the skip connections to 0 for performance gain!!
    def __init__(self,
                 c_list: list[int],
                 spatial_factor_list: list[int],
                 num_blocks_list: list[int],
                 c_in: int,
                 c_target: int = 64,
                 height_width_target: int = 32,
                 ):
        super().__init__()
        self.c_in = c_in
        if not len(c_list) - 1 == len(spatial_factor_list) == len(num_blocks_list):
            raise ValueError('The length of c_list - 1 and  length of spatial_factor_list have to be equal as they correspond to'
                             'the number of downscalings in our network')
        self.upscale_input_channels = nn.Conv2d(c_in, c_list[0], kernel_size=1)
        self.encoder = Encoder(c_list, spatial_factor_list, num_blocks_list)
        self.decoder = Decoder(c_list, spatial_factor_list, num_blocks_list)
        self.center_crop_downscale = DownScaleToTarget(c_in=c_list[0], c_out=c_target, height_width_target=height_width_target)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        if x.shape[1] != self.c_in:
            raise ValueError(
                f'Size of channel dim in network input is {x.shape[1]} but expected {self.c_in}'
            )
        x = self.upscale_input_channels(x)
        skip_list = self.encoder(x)
        x = self.decoder(skip_list)
        x = self.center_crop_downscale(x)

        # XEntropy takes non-softmaxed input!
        # x = self.soft_max(x)
        return x
