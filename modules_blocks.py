import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from helper_functions import create_dilation_list


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
        # T.CenterCrop(size=self.width_height_out)(x)
        x = self.conv(x)
        return x


# class MetResModule(nn.Module):
#     """
#     Residual module inspired by MetNet2
#     channel num (c_num) and width height remain conserved
#     Applies dilated convolution starting with a dilation rate of 1 (normal convolution) exponentially building up to a
#     dilation rate that results in a virtual kernel size of 1/2 (or whatever is given by inverse_ratio)
#     of the width and height
#     """
#     def __init__(self, c_num: int, width_height: int, kernel_size: int, num_dils: int, stride: int=1, inverse_ratio=2):
#         super().__init__()
#         # TODO: Implement this with nn.ModuleDict() in future!!
#         self.dilation_list = create_dilation_list(width_height, inverse_ratio=inverse_ratio)
#         self.dil_block1 = MetDilBlock(c_num, width_height, self.dilation_list[0], kernel_size, stride)
#         self.dil_block2 = MetDilBlock(c_num, width_height, self.dilation_list[1], kernel_size, stride)
#         self.dil_block3 = MetDilBlock(c_num, width_height, self.dilation_list[2], kernel_size, stride)
#         self.dil_block4 = MetDilBlock(c_num, width_height, self.dilation_list[3], kernel_size, stride)
#         self.dil_block5 = MetDilBlock(c_num, width_height, self.dilation_list[4], kernel_size, stride)
#         self.dil_block6 = MetDilBlock(c_num, width_height, self.dilation_list[5], kernel_size, stride)
#
#     def forward(self, x: torch.Tensor):
#
#         out = self.dil_block1(x)
#         out = self.dil_block2(out)
#         out = self.dil_block3(out)
#         out = self.dil_block4(out)
#         out = self.dil_block5(out)
#         out = self.dil_block6(out)
#         return out


class MetResModule(nn.Module):
    """
    Residual module inspired by MetNet2
    channel num (c_num) and width height remain conserved
    Applies dilated convolution starting with a dilation rate of 1 (normal convolution) exponentially building up to a
    dilation rate that results in a virtual kernel size of 1/2 (or whatever is given by inverse_ratio)
    of the width and height
    """
    def __init__(self, c_num: int, width_height: int, kernel_size: int, num_dils: int, stride: int=1, inverse_ratio=2):
        super().__init__()
        # TODO: Implement this with nn.ModuleDict() in future!!
        self.dilation_list = create_dilation_list(width_height, inverse_ratio=inverse_ratio)
        self.dilation_blocks = nn.ModuleList()
        # Create the amount of dilation blocks needed to increase dilation factor exponentially until kernel reaches
        # size defined by inverse_ratio
        for i, dilation in enumerate(self.dilation_list):
            self.dilation_blocks.add_module(
                MetDilBlock(c_num, width_height, dilation, kernel_size, stride)
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
    def __init__(self, c_num: int, width_height: int, dilation: int, kernel_size: int, stride: int=1):
        super().__init__()
        # TODO implement formula

        padding = 'same'
        self.dilation1 = nn.Conv2d(c_num, c_num, kernel_size, dilation=dilation, stride=stride, padding=padding)
        self.layer_norm1 = nn.LayerNorm(width_height)
        self.dilation2 = nn.Conv2d(c_num, c_num, kernel_size, dilation=dilation, stride=stride, padding=padding)
        self.layer_norm2 = nn.LayerNorm(width_height)

        # self.layernorm = nn.LayerNorm() needs w x h dimensions as input

    def forward(self, x: torch.Tensor):

        out = x
        out = self.dilation1(out)
        out = self.layer_norm1(out)
        out = F.relu(out)
        out = self.dilation2(out)
        out = self.layer_norm2(out)
        out = F.relu(out)

        return x + out



