import einops
import torch
import torch.nn.functional as F
import torch.nn as nn
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
    def __init__(self, c: int, c_factor: int = 2, spacial_factor: int = 2):
        super().__init__()
        #Does the conv 2 2 with stride 2 cause image patching? If so try conv 3x3 with padding
        self.conv = nn.Conv2d(c, c * c_factor, kernel_size=spacial_factor, stride=spacial_factor)
        self.layer_norm = LayerNormChannelOnly(c * c_factor)
        self.conv_skip = nn.Conv2d(c, c * c_factor, kernel_size=1)

    def forward(self, x: torch.Tensor):
        skip = x
        skip = self.conv_skip(skip)

        x = self.conv(x)
        x = self.layer_norm(x)

        x += skip
        return x


class ConvNextUpScale(nn.Module):
    """
    Upscaling ConvNext block
    """

    def __init__(self, c: int, c_factor: int = 2, spacial_factor: int = 2):
        super().__init__()
        if c % c_factor != 0:
            raise ValueError('Channel number has to divisible by scale_factor.')
        # Does the conv 2 2 with stride 2 cause image patching? If so try conv 3x3 with padding
        self.conv_transposed = nn.ConvTranspose2d(c, c / c_factor, kernel_size=spacial_factor, stride=spacial_factor)
        self.layer_norm = LayerNormChannelOnly(c / c_factor)
        self.conv_skip = nn.Conv2d(c, c / c_factor, kernel_size=1)

    def forward(self, x: torch.Tensor):
        skip = x
        skip = self.conv_skip(skip)

        x = self.conv_transposed(x)
        x = self.layer_norm(x)

        x += skip
        return x


class EncoderModule(nn.Module):
    """
    One block of the Encoder that includes two ConvNext Block and a downsampling block
    """
    # Potentially implement this with nn.Sequencial to make number of blocks per module as well as c_in and c_out variable
    def __init__(self, c: int):
        super().__init__()
        self.conv_next_1 = ConvNext(c)
        self.conv_next_2 = ConvNext(c)
        self.conv_next_down_scale = ConvNextDownScale(c)

    def forward(self, x: torch.Tensor):
        x = self.conv_next_1(x)
        x = self.conv_next_2(x)
        x = self.conv_next_down_scale(x)

class DecoderModule(nn.Module):
    """
    One block of the Decoder that includes two ConvNext Block and an upsampling block
    """
    def __init__(self, c: int):
        super().__init__()
        self.conv_next_1 = ConvNext(c)
        self.conv_next_2 = ConvNext(c)
        self.conv_next_up_scale = ConvNextUpScale(c)

    def forward(self, x: torch.Tensor):
        x = self.conv_next_1(x)
        x = self.conv_next_2(x)
        x = self.conv_next_up_scale(x)


class Encoder(nn.Module):
    """
    Encoder
    This returns the whole list for the skip connections
    """
    def __init__(self, c_in: int, num_downscales: int):
        super().__init__()

        # self.conv_next_1 = ConvNext(c_in)
        # self.conv_next_2 = ConvNext(c_in)

        self.module_list = nn.ModuleList()
        c_curr = c_in
        for i in range(num_downscales):
            # 256 / 2 ** 5 = 8
            i += 1
            self.module_list.add_module(
                name=r'encoder_module_{i}',
                module=EncoderModule(c=c_curr)
            )

    def forward(self, x: torch.Tensor):
        skip_list = []
        skip_list.append(x)  # A skip connection right from the input to the output

        for module in self.net_modules(x):
            x = module(x)
            skip_list.append(x)
        return skip_list


class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, c: int, num_upscales: int,):

        self.module_list = nn.ModuleList()
        c_curr = c
        for i in range(num_upscales):
            # 256 / 2 ** 5 = 8
            i += 1
            self.module_list.add_module(
                name=r'encoder_module_{i}',
                module=EncoderModule(c=c_curr)
            )

    def forward(self, skip_list: list):
        x = skip_list[0]
        for module, skip in zip(self.module_list, reversed(skip_list[1:])):
            x = module(x)
            x += skip
        return x


class UNet(nn.Module):
    def __init__(self, c_in,):
        num_scalings = 4
        self.encoder = Encoder(c_in, num_scalings)
        c_latent = c_in * 2 ** num_scalings
        self.decoder = Decoder(c_latent, num_scalings)

    def forward(self, x: torch.Tensor):
        skip_list = self.encoder(x)
        x = self.decoder(skip_list)
        return x





