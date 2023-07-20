import torch
import numpy as np
import scipy

def convolution_no_channel_sum(input, kernel, device, center_crop_size=32):
    '''
    This is an implementation of a convolution operation using a single filter (iterated only once per b dim), that skips
    summing the results of all channels. This way the output is not a channel dim of size 1 but instead the channel dim is preserved

    center_crop_size determines the size of the output.
    This function expects a bigger input than output with respect to h and w
    , where the input has to be bigger by at least input.shape[-1] + kernel.shape[-1]
    This is done such, that the kernel can also be implemented on all values outside the edges of the produced target.

    Output has conserved batch dim, conserved channel dim (reason for this funcrion), center_crop_size, center_crop_size

    '''

    start_height, start_width, end_height, end_width = get_center_crop_indices(input, center_crop_size)
    kernel_size = kernel.shape[-1]

    # TODO: Speed this up with jit compiler (but then torch cannot be used)??!!
    output = torch.zeros(input.shape[0], input.shape[1], center_crop_size, center_crop_size, device=device)
    for h in  range(center_crop_size):
        h_extended = h + start_height
        for w in range(center_crop_size):
            w_extended = w + start_width

            h_start = h_extended - kernel_size // 2
            h_end = h_start + kernel_size
            w_start = w_extended - kernel_size // 2
            w_end = w_start + kernel_size

            vals = input[:, :, h_start : h_end, w_start : w_end] * kernel
            output[:, :, h, w] = torch.sum(vals, dim=(2,3)) # Only sum along h and w, KEEP c dim
            # (this is why we nee to rewrite this) and
            # of course keep b dim

    return output


def gaussian_smoothing_target(target_one_hot_extended, sigma, kernel_size, device, target_size=32):
    kernel_shape = (target_one_hot_extended.shape[0], target_one_hot_extended.shape[1], kernel_size, kernel_size)
    kernel = create_gaussian_kernel(kernel_shape, sigma, device)

    target_one_hot = convolution_no_channel_sum(target_one_hot_extended, kernel, device)
    return target_one_hot


def create_gaussian_kernel(shape, sigma, device):
    '''
    Create a gaussian kernel along w and h , with the same gaussian filter along the b and c dimensions
    '''
    n = np.zeros((shape[2], shape[3]))
    n[int(shape[2]//2) , int(shape[3]//2)] = 1 # This selects center exactly (tested)
    gauss_2d = scipy.ndimage.gaussian_filter(n, sigma=sigma)

    kernel = np.zeros(shape)
    for b_i in range(shape[0]):
        for c_i in range(shape[1]):
            kernel[b_i, c_i, :, :] = gauss_2d

    return torch.from_numpy(kernel).to(device)


def get_center_crop_indices(image, crop_size):
    _, _, height, width = image.size()  # Get the height and width of the image

    # Calculate the starting indices for the center crop
    start_height = (height - crop_size) // 2
    start_width = (width - crop_size) // 2

    # Calculate the ending indices for the center crop
    end_height = start_height + crop_size
    end_width = start_width + crop_size

    return start_height, start_width, end_height, end_width
