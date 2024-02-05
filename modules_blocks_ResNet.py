import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from helper.helper_functions import create_dilation_list


# TODO: This Resnet has 5 Residual Blocks. ResNet50 has 16 residual blocks. ResNet101 has 33 residual blocks
class ResNet(nn.Module):
    def __init__(self, c_in: int, s_upscale_c_to, s_num_bins_crossentropy, s_width_height: int,
                    s_gaussian_smoothing_multiple_sigmas, s_multiple_sigmas, **__):
        """
        ResNet
        The initial 7 x 7 layer has been skipped
        s_upscale_c_to: (deactivated) the number of channels, that the first convolution scales up to
        """
        super().__init__()
        # downsample_at = [3, 6, 12] # For Resnet 34
        downsample_at = [25, 50, 100]
        s_width_height_in = s_width_height
        self.conv1_1_upscale = nn.Conv2d(c_in, s_upscale_c_to, kernel_size=1, dilation=1, stride=1, padding=0)
        self.net_modules = nn.ModuleList()
        self.soft_max = nn.Softmax(dim=1)
        c_curr = c_in
        # c_curr = s_upscale_c_to
        for i in range(150):
            # We have to downsample 3 times to get from height of 256 to 32
            i += 1
            if i in downsample_at:
                downsample = True
                downsample_str = '_downsample'
            else:
                downsample = False
                downsample_str = ''

            # self.net_modules.add_module(
            #     name='resnet_module_{}_{}'.format(i, downsample_str),
            #     module=ResNetBlock(c_in=c_curr, downsample=downsample)
            # )

            self.net_modules.add_module(
                name='convnext_module_{}_{}'.format(i, downsample_str),
                module=ConvNeXtBlock(c_in=c_curr, downsample=downsample)
            )

            if downsample:
                c_curr = c_curr * 2
        if not s_gaussian_smoothing_multiple_sigmas:
            downscale_c_to = s_num_bins_crossentropy
        else:
            downscale_c_to = s_num_bins_crossentropy * len(s_multiple_sigmas)

        self.conv1_1_downscale = nn.Conv2d(c_curr, downscale_c_to, kernel_size=1, dilation=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        # x = self.conv1_1_upscale(x)
        for module in self.net_modules:
            x = module(x)
        x = self.conv1_1_downscale(x)
        x = self.soft_max(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, c_in: int, downsample: bool, kernel_size=3):
        super().__init__()
        if downsample:
            stride = 2
            c_out = c_in * 2
        else:
            stride = 1
            c_out = c_in

        if kernel_size % 2 == 0:
            # For even-sized kernel, pad equally on both sides
            padding = kernel_size // 2
        else:
            # For odd-sized kernel, pad using the formula (kernel_size - 1) // 2
            padding = (kernel_size - 1) // 2

        self.downsample = downsample
        # Only conv1 downsamples
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, dilation=1, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=kernel_size, dilation=1, stride=1, padding=padding)
        self.conv_res_connection = nn.Conv2d(c_in, c_out, kernel_size=1, dilation=1, stride=stride, padding=0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(c_out)
        self.bn2 = nn.BatchNorm2d(c_out)


    def forward(self, x: torch.Tensor):
        '''
        nicely described in: https://www.youtube.com/watch?v=o_3mboe1jYI&t=324s
        https://wisdomml.in/understanding-resnet-50-in-depth-architecture-skip-connections-and-advantages-over-other-networks/
        '''
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        # Res connection has 1x1 conv for down sampling to fit channel dimensions
        if self.downsample:
            res = self.conv_res_connection(res)
        x = res + x
        x = self.relu2(x)
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, c_in: int,  downsample: bool):
        super().__init__()

        self.downsample = downsample
        c_in_of_block = c_in
        # Conv 1 --> 7x7 depth wise convolution
        c_out = c_in * 4
        if downsample:
            c_out *= 2
            stride = 2
        else:
            stride = 1
        kernel_size_1 = 7
        padding_1 = (kernel_size_1 - 1) // 2
        # Changing c_in to c_out by a factor K and groups = c_out convolution becomes depth wise convolution
        # (see torch docu)
        self.d_conv1 = nn.Conv2d(c_in, c_out, kernel_size=kernel_size_1, stride=stride, padding=padding_1,
                                groups=c_in)

        # TODO: implement Layernorm which requires dimensionality as input
        # self.layer_norm = nn.Layernorm()
        self.batch_norm = nn.BatchNorm2d(c_out)

        #Conv 2 --> 1x1 conv
        c_in = c_out
        c_out = c_in // 4
        self.conv2 = nn.Conv2d(c_in, c_out, kernel_size=1)

        self.gelu = nn.GELU()

        #Conv 3 --> 1x1 conv
        c_in = c_out
        self.conv3 = nn.Conv2d(c_in, c_out, kernel_size=1)

        self.conv_res_connection = nn.Conv2d(c_in_of_block, c_out, stride=2, kernel_size=1)

    def forward(self, x: torch.Tensor):
        res = x
        x = self.d_conv1(x)
        # TODO: originally layernorm
        x = self.batch_norm(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        # Res connection has 1x1 conv for down sampling to fit channel dimensions
        if self.downsample:
            res = self.conv_res_connection(res)
        x += res
        return x





