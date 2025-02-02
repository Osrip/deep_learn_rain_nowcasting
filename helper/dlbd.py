import torch
import torch.nn.functional as F


def dlbd_traget_pre_processing(input_tensor, output_size, sigma=1.0, kernel_size=None):
    """
    Sped up version using two 1D convolutions (vertical then horizontal) that always produces the desired output size.

    When kernel_size is None, the maximum allowed odd kernel is used so that no cropping is needed.
    When a manual (odd) kernel_size is provided (and it's smaller than the maximum), the input is center-cropped
    so that the valid convolution produces exactly the desired output size. This can save compute.

    Args:
        input_tensor (torch.Tensor): Shape (B, C, H, W), one-hot.
        output_size (int or tuple): Desired output size (int or (desired_H, desired_W)); must be smaller than input.
        sigma (float): Standard deviation of the Gaussian.
        kernel_size (int or tuple, optional): Manual kernel size. Must be <= maximal kernel size
            (computed as H - desired_H + 1 and W - desired_W + 1) and must be odd.
            If None, kernel sizes are computed automatically.

    Returns:
        torch.Tensor: Output after convolution with shape (B, C, desired_H, desired_W).
    """
    input_tensor = input_tensor.float()
    B, C, H, W = input_tensor.shape

    if isinstance(output_size, int):
        desired_H = desired_W = output_size
    else:
        desired_H, desired_W = output_size

    assert desired_H < H and desired_W < W, "Desired output size must be smaller than input dimensions."

    # Maximum allowed kernel sizes for valid convolution:
    max_kernel_h = H - desired_H + 1
    max_kernel_w = W - desired_W + 1

    if kernel_size is None:
        # Use the maximum allowed odd kernel sizes.
        kernel_h = max_kernel_h if max_kernel_h % 2 == 1 else max_kernel_h - 1
        kernel_w = max_kernel_w if max_kernel_w % 2 == 1 else max_kernel_w - 1
        # No cropping needed: (H - kernel_h + 1 == desired_H)
    else:
        # Use manual kernel size.
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        elif isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2:
            kernel_h, kernel_w = kernel_size
        else:
            raise ValueError("kernel_size must be an int or a tuple/list of two ints.")

        if kernel_h > max_kernel_h or kernel_w > max_kernel_w:
            raise ValueError(f"Manual kernel size ({kernel_h}, {kernel_w}) exceeds the maximum allowed "
                             f"({max_kernel_h}, {max_kernel_w}).")
        if kernel_h % 2 == 0 or kernel_w % 2 == 0:
            raise ValueError(f"Kernel sizes must be odd numbers. Got ({kernel_h}, {kernel_w}).")
        # Center-crop the input so that after valid convolution with manual kernel the output size is desired.
        crop_H = desired_H + kernel_h - 1  # because output height = cropped_H - kernel_h + 1
        crop_W = desired_W + kernel_w - 1
        offset_H = (H - crop_H) // 2
        offset_W = (W - crop_W) // 2
        input_tensor = input_tensor[:, :, offset_H:offset_H + crop_H, offset_W:offset_W + crop_W]
        # Update H, W for clarity (not strictly needed)
        H, W = crop_H, crop_W

    # Create vertical 1D Gaussian kernel (shape: (kernel_h, 1))
    coords_h = torch.arange(kernel_h, device=input_tensor.device, dtype=input_tensor.dtype) - kernel_h // 2
    g_1d_h = torch.exp(-(coords_h ** 2) / (2 * sigma ** 2))
    g_1d_h /= g_1d_h.sum()
    # Reshape to (C, 1, kernel_h, 1) for depthwise convolution
    g_1d_h = g_1d_h.view(1, 1, kernel_h, 1).repeat(C, 1, 1, 1)

    # Convolve vertically (valid convolution)
    out = F.conv2d(input_tensor, g_1d_h, groups=C, padding=0)

    # Create horizontal 1D Gaussian kernel (shape: (1, kernel_w))
    coords_w = torch.arange(kernel_w, device=input_tensor.device, dtype=input_tensor.dtype) - kernel_w // 2
    g_1d_w = torch.exp(-(coords_w ** 2) / (2 * sigma ** 2))
    g_1d_w /= g_1d_w.sum()
    # Reshape to (C, 1, 1, kernel_w) for depthwise convolution
    g_1d_w = g_1d_w.view(1, 1, 1, kernel_w).repeat(C, 1, 1, 1)

    # Convolve horizontally (valid convolution)
    out = F.conv2d(out, g_1d_w, groups=C, padding=0)
    return out