import einops
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from helper.helper_functions import create_dilation_list


class Conv3x3(nn.Module):
    """
    Performs conv 3x3 and LeakyReLu
    c: num of channels
    num of input and output channels are equal
    """
    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        #TODO in original implementation they use ReLu non-leaky
        x = F.leaky_relu(x)
        return x


class EncoderModule(nn.Module):
    """
    Encoder down sampling module of the UNet

    Has subsequent Conv3x3 convolutions (given by num_convs)
     and a subsequent conv down scaling (spatial down scaling and channel upscaling)

    Args:
        c_in: number of input channels
        c_out: number of output channels
        spatial_factor: downscaling factor (stride and kernel size of convolution)
        num_convs: number of subsequent Conv3x3 (not spatially scaling)

    """
    def __init__(self, c_in: int, c_out: int, spatial_factor: int, num_convs: int):
        super().__init__()

        # Each block consists of 'num_convs' subsequent conv3x3 and a down scaling block
        self.conv_blocks = nn.Sequential(*[Conv3x3(c_out) for _ in range(num_convs)])
        self.conv_down_scale = nn.Conv2d(c_in, c_out, kernel_size=spatial_factor, stride=spatial_factor)

    def forward(self, x: torch.Tensor):
        x = self.conv_blocks(x)
        x = self.conv_down_scale(x)
        return x


class DecoderModule(nn.Module):
    """
    Decoder up sampling module of the UNet

    Has a conv up scaling (spatial up scaling and channel up scaling)
     and subsequent Conv3x3 convolutions (given by num_convs)

    Args:
        c_in: number of input channels
        c_out: number of output channels
        spatial_factor: downscaling factor (stride and kernel size of convolution)
        num_convs: number of subsequent Conv3x3 (not spatially scaling)
    """
    def __init__(self, c_in: int, c_out: int, spatial_factor: int, num_convs: int):
        super().__init__()
        self.conv_next_up_scale = nn.Conv2d(c_out, c_in, kernel_size=spatial_factor, stride=spatial_factor)
        self.conv_next_blocks = nn.Sequential(*[Conv3x3(c_out) for _ in range(num_convs)])

    def forward(self, x: torch.Tensor):
        x = self.conv_next_up_scale(x)
        x = self.conv_next_blocks(x)
        return x


class Encoder(nn.Module):
    """
    Encoder
    This returns the whole list for the skip connections
    """
    def __init__(self, c_list: list[int], spatial_factor_list: list[int], num_conv_list: list[int]):
        super().__init__()

        self.module_list = nn.ModuleList()

        for i, (c_in, c_out, spatial_factor, num_convs) in enumerate(
                zip(
                    c_list[:-1],
                    c_list[1:],
                    spatial_factor_list,
                    num_conv_list
                )
        ):
            i += 1
            self.module_list.add_module(
                name=f'encoder_module_{i}',
                module=EncoderModule(c_in, c_out, spatial_factor, num_convs)
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
    """
    def __init__(self, c_list: list[int], spatial_factor_list: list[int], num_conv_list: list[int]):
        super().__init__()

        self.module_list = nn.ModuleList()

        reversed_c_list = list(reversed(c_list))
        reversed_spatial_factors = list(reversed(spatial_factor_list))
        reversed_num_convs = list(reversed(num_conv_list))

        for i, (c_in, c_out, spatial_factor, num_convs) in enumerate(
                zip(
                    reversed_c_list[:-1],
                    reversed_c_list[1:],
                    reversed_spatial_factors,
                    reversed_num_convs
                )
        ):
            i += 1
            self.module_list.add_module(
                name=f'decoder_module_{i}',
                module=DecoderModule(c_in, c_out, spatial_factor, num_convs)
            )

    def forward(self, skip_list: list):
        x = skip_list[-1]  # Start with the first element from the skip connections
        reversed_skips = reversed(skip_list[:-1])
        for module, skip in zip(self.module_list, reversed_skips):
            x = module(x)
            x += skip  # Add the skip connection to the output of the current module
        return x


class UNet(nn.Module):
    """
    Unet
    c_in: Number of input channels (is up scaled to c_list[0]), < c_list[0]
    c_out: Number of output channels (is down scaled to from c_list[-1]), <c_list[0]

    c_list: list that includes the number of channels that each scaling module should input / output
            The length of the list corresponds to the (number of down scalings = number of upscalings) + 1
            in the UNet.
    spatial_factor_list: list that includes the spatial down scaling / up scaling factors for each down / up scaling
            module. The length of the list corresponds to the number of down scalings = number of up scalings
    num_conv_list: List of the number of Conv3x3 + Leaky ReLu for each down scaling / up scaling.
            The spatial down scaling / ups scaling block itself is not included in the number.
            The length of the list corresponds to the number of down scalings = number of ups calings

    Properties:
        Assumes same spatial dimensions for input and ouptut
        Skip Connection right from input to output

        The number of encoder and decoder modules is given

    Example usage:
        self.model = UNet(
            c_list=[4, 32, 64, 128, 256],
            spatial_factor_list=[2, 2, 2, 2],
            num_conv_list=[2, 2, 2, 2],
            )
    """
    def __init__(
            self,
            c_list: list[int],
            spatial_factor_list: list[int],
            num_conv_list: list[int],
            c_in: int,
            c_out: int,
    ):
        super().__init__()
        self.c_in = c_in
        if not len(c_list) - 1 == len(spatial_factor_list) == len(num_conv_list):
            raise ValueError('The length of c_list - 1 and  length of spatial_factor_list have to be equal as they correspond to'
                             'the number of downscalings in our network')

        self.upscale = nn.Conv2d(c_in, c_list[0], kernel_size=1)
        self.encoder = Encoder(c_list, spatial_factor_list, num_conv_list)
        self.decoder = Decoder(c_list, spatial_factor_list, num_conv_list)
        self.downscale = nn.Conv2d(c_list[-1], c_out, kernel_size=1)

    def forward(self, x: torch.Tensor):
        if x.shape[1] != self.c_in:
            raise ValueError(
                f'Size of channel dim in network input is {x.shape[1]} but expected {self.c_in}'
            )

        x = self.upscale(x)
        skip_list = self.encoder(x)
        x = self.decoder(skip_list)
        x = self.downscale(x)
        return x
