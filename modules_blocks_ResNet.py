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
        s_upscale_c_to: the number of channels, that the first convolution scales up to
        """
        super().__init__()
        s_width_height_in = s_width_height
        self.conv1_1_upscale = nn.Conv2d(c_in, s_upscale_c_to, kernel_size=1, dilation=1, stride=1, padding=0)
        self.net_modules = nn.ModuleList()
        self.soft_max = nn.Softmax(dim=1)
        test_list = []
        c_curr = s_upscale_c_to
        for i in range(5):
            # We have to downsample 3 times to get from height of 256 to 32
            i += 1
            if i % 2 == 1:
                downsample = True
            else:
                downsample = False

            self.net_modules.add_module(
                name='resnet_module_{}'.format(i),
                module=ResNetBlock(c_in=c_curr, kernel_size=3, downsample=downsample)
            )
            if downsample:
                c_curr = c_curr * 2
        if not s_gaussian_smoothing_multiple_sigmas:
            downscale_c_to = s_num_bins_crossentropy
        else:
            downscale_c_to = s_num_bins_crossentropy * len(s_multiple_sigmas)

        self.conv1_1_downscale = nn.Conv2d(c_curr, downscale_c_to, kernel_size=1, dilation=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x = self.conv1_1_upscale(x)
        for module in self.net_modules:
            x = module(x)
        x = self.conv1_1_downscale(x)
        x = self.soft_max(x)
        return x



class ResNetBlock(nn.Module):
    def __init__(self, c_in: int, kernel_size: int, downsample: bool):
        '''
        For
        '''
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
        TODO: Implement properly from:
        https://wisdomml.in/understanding-resnet-50-in-depth-architecture-skip-connections-and-advantages-over-other-networks/
        '''
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        # Res connection has 1x1 conv to fit channel dimensions
        if self.downsample:
            x = self.conv_res_connection(x)
        out = x + out
        out = self.relu2(out)
        return out