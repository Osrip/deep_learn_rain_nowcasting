import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from helper_functions import create_dilation_list


class Network(nn.Module):
    def __init__(self, c_in: int, upscale_c_to, num_bins_crossentropy, width_height_in: int):
        """
        upscale_c_to: the number of channels, that the first convolution scales up to
        """
        super().__init__()
        # TODO No doubling of channels in conv 2d???!!!
        self.conv1_1_upscale = nn.Conv2d(c_in, upscale_c_to, kernel_size=1, dilation=1, stride=1, padding=0)
        self.net_modules = nn.ModuleList()
        self.soft_max = nn.Softmax(dim=1)
        test_list = []
        c_curr = upscale_c_to
        for i in range(3):
            # log2(256)=8 log2(32)=5 --> We need three sampling modules that half the size
            # ['c_in: 16 c_out: 32 curr_height: 256.0 out_height: 128.0',
            # 'c_in: 32 c_out: 64 curr_height: 128.0 out_height: 64.0',
            # 'c_in: 64 c_out: 128 curr_height: 64.0 out_height: 32.0']
            i += 1
            self.net_modules.add_module(
                name='res_module_{}'.format(i),
                module=MetResModule(c_num=c_curr, width_height=int(width_height_in / (2 ** (i-1))), kernel_size=3,
                                    stride=1, inverse_ratio=2)
            )
            self.net_modules.add_module(
                name='sample_module_{}'.format(i),
                module=SampleModule(c_in=c_curr, c_out=c_curr * 2, width_height_out=int(width_height_in / (2 ** i)))  # w h
            )
            c_curr = c_curr * 2

            test_list.append('c_in: {} c_out: {} curr_height: {} out_height: {}'
                             .format(c_curr, c_curr * 2, width_height_in / (2 ** (i-1)), width_height_in / (2 ** i)))

        self.conv1_1_downscale = nn.Conv2d(c_curr, num_bins_crossentropy, kernel_size=1, dilation=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        x = self.conv1_1_upscale(x)
        for module in self.net_modules:
            x = module(x)
        x = self.conv1_1_downscale(x)
        x = self.soft_max(x)
        return x


class SampleModule(nn.Module):
    """
    Sample Module
    Crops the picture (recommended crop to 1/2 height_width) and upsamples channels (recommended to 2 x c_in)
    """
    def __init__(self, c_in: int, c_out: int, width_height_out: int):
        super().__init__()
        self.crop = T.CenterCrop(size=width_height_out)
        # self.width_height_out = width_height_out
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
    def __init__(self, c_num: int, width_height: int, kernel_size: int = 3, stride: int = 1, inverse_ratio=2):
        super().__init__()
        self.dilation_list = create_dilation_list(width_height, inverse_ratio=inverse_ratio)
        self.dilation_blocks = nn.ModuleList()
        # Create the amount of dilation blocks needed to increase dilation factor exponentially until kernel reaches
        # size defined by inverse_ratio
        for i, dilation in enumerate(self.dilation_list):
            self.dilation_blocks.add_module(
                name='dil_block_{}'.format(i), module=MetDilBlock(c_num, width_height, dilation, kernel_size, stride)
            )

    def forward(self, x: torch.Tensor):
        for block in self.dilation_blocks:
            x = block(x)
        return x


class MetDilBlock(nn.Module):
    """
    The Dilation Block. Similar to MetNet2's Dilation Block
    height, width reimain conserved, chanell number remains conserved
    """
    def __init__(self, c_num: int, width_height: int, dilation: int, kernel_size: int, stride: int = 1):
        super().__init__()
        # TODO implement formula

        padding = 'same'
        self.dilation1 = nn.Conv2d(c_num, c_num, kernel_size, dilation=dilation, stride=stride, padding=padding)
        self.layer_norm1 = nn.LayerNorm(width_height)
        self.dilation2 = nn.Conv2d(c_num, c_num, kernel_size, dilation=dilation, stride=stride, padding=padding)
        self.layer_norm2 = nn.LayerNorm(width_height)

    def forward(self, x: torch.Tensor):

        out = x
        out = self.dilation1(out)
        out = self.layer_norm1(out)
        out = F.relu(out)
        out = self.dilation2(out)
        out = self.layer_norm2(out)
        out = F.relu(out)

        return x + out