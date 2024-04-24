import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from helper.helper_functions import create_dilation_list


class Network(nn.Module):
    def __init__(self, c_in: int, s_upscale_c_to, s_num_bins_crossentropy, s_width_height: int,
                 s_gaussian_smoothing_multiple_sigmas, s_multiple_sigmas, **__):
        """
        s_upscale_c_to: the number of channels, that the first convolution scales up to
        """
        super().__init__()
        s_width_height_in = s_width_height

        # TODO No doubling of channels in conv 2d???!!!
        self.conv1_1_upscale = nn.Conv2d(c_in, s_upscale_c_to, kernel_size=1, dilation=1, stride=1, padding=0)
        self.net_modules = nn.ModuleList()
        self.soft_max = nn.Softmax(dim=1)
        test_list = []
        c_curr = s_upscale_c_to
        for i in range(3):
            # log2(256)=8 log2(32)=5 --> We need three sampling modules that half the size
            # ['c_in: 16 c_out: 32 curr_height: 256.0 out_height: 128.0',
            # 'c_in: 32 c_out: 64 curr_height: 128.0 out_height: 64.0',
            # 'c_in: 64 c_out: 128 curr_height: 64.0 out_height: 32.0']
            i += 1
            self.net_modules.add_module(
                name='res_module_{}'.format(i),
                module=MetResModule(c_num=c_curr, s_width_height=int(s_width_height_in / (2 ** (i-1))), kernel_size=3,
                                    stride=1, inverse_ratio=2)
            )
            self.net_modules.add_module(
                name='sample_module_{}'.format(i),
                module=SampleModule(c_in=c_curr, c_out=c_curr * 2, s_width_height_out=int(s_width_height_in / (2 ** i)))  # w h
            )
            c_curr = c_curr * 2

            test_list.append('c_in: {} c_out: {} curr_height: {} out_height: {}'
                             .format(c_curr, c_curr * 2, s_width_height_in / (2 ** (i-1)), s_width_height_in / (2 ** i)))

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
        # TODO: SOFTMAX SCHON IN X ENTROPY??
        x = self.soft_max(x)
        return x


class SampleModule(nn.Module):
    """
    Sample Module
    Crops the picture (recommended crop to 1/2 height_width) and upsamples channels (recommended to 2 x c_in)
    """
    def __init__(self, c_in: int, c_out: int, s_width_height_out: int):
        super().__init__()
        self.crop = T.CenterCrop(size=s_width_height_out)
        # self.s_width_height_out = s_width_height_out
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, dilation=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x = self.crop(x)
        x = self.conv(x)
        return x


class MetResModule(nn.Module):
    """
    Residual module inspired by MetNet2
    channel num (c_num) and width height remain conserved
    Applies dilated convolution starting with a dilation rate of 1 (normal convolution) exponentially building up to a
    dilation rate that results in a virtual kernel size of 1/2 (or whatever is given by inverse_ratio)
    of the width and height
    """
    def __init__(self, c_num: int, s_width_height: int, kernel_size: int = 3, stride: int = 1, inverse_ratio=2):
        super().__init__()
        self.dilation_list = create_dilation_list(s_width_height, inverse_ratio=inverse_ratio)
        self.dilation_blocks = nn.ModuleList()
        # Create the amount of dilation blocks needed to increase dilation factor exponentially until kernel reaches
        # size defined by inverse_ratio
        for i, dilation in enumerate(self.dilation_list):
            self.dilation_blocks.add_module(
                name='dil_block_{}'.format(i), module=MetDilBlock(c_num, s_width_height, dilation, kernel_size, stride)
            )

    def forward(self, x: torch.Tensor):
        for block in self.dilation_blocks:
            x = block(x)
        return x


class MetDilBlock(nn.Module):
    """
    The Dilation Block. Similar to MetNet2's Dilation Block
    height, width reimain conserved, channel number remains conserved
    """
    def __init__(self, c_num: int, s_width_height: int, dilation: int, kernel_size: int, stride: int = 1):
        super().__init__()
        # TODO implement formula

        padding = 'same'
        self.dilation1 = nn.Conv2d(c_num, c_num, kernel_size, dilation=dilation, stride=stride, padding=padding)
        self.group_norm1 = nn.GroupNorm(1, c_num)
        self.dilation2 = nn.Conv2d(c_num, c_num, kernel_size, dilation=dilation, stride=stride, padding=padding)
        self.group_norm2 = nn.GroupNorm(1, c_num)

    def forward(self, x: torch.Tensor):

        out = x
        out = self.dilation1(out)
        out = self.group_norm1(out)
        out = F.relu(out)
        out = self.dilation2(out)
        out = self.group_norm2(out)
        out = F.relu(out)

        return x + out