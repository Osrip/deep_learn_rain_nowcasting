import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec


def plot_distributions(target: torch.Tensor, pred_binned: torch.Tensor,
                               linspace_binning_inv_norm: np.ndarray, linspace_binning_max_inv_norm: float,
                               title, save_path_name, num_bar_plots=10, **__):
    """
    Plot distributions for num_bar_plots pixels sampled uniformly from the diagonal of the picture.

    Args:
    - target (torch.Tensor): Target image, shape (b, 32, 32).
    - pred_binned (torch.Tensor): Predicted image, shape (b, 64, 32, 32).
    - linspace_binning_inv_norm (np.ndarray): Left sides of bins in log space.
    - linspace_binning_max_inv_norm (float): Rightmost bin value in log space.
    - title (str): Title of the plot.
    - num_bar_plots (int): Number of pixels to sample.
    """
    target = target.detach().cpu().numpy()
    pred_binned = pred_binned.detach().cpu().numpy()

    b, _, h, w = pred_binned.shape
    fig = plt.figure(figsize=(10, num_bar_plots * 2))
    gs = GridSpec(num_bar_plots, 1, figure=fig)

    # Sampling num_bar_plots pixels from the diagonal
    indices = np.linspace(0, h - 1, num_bar_plots, dtype=int)
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(gs[i, 0])
        pixel_distribution = pred_binned[0, :, idx, idx]  # Assuming b = 0 for simplicity

        # Constructing bin edges
        bin_edges = np.append(linspace_binning_inv_norm, linspace_binning_max_inv_norm)

        # Plotting the distribution as a bar plot
        ax.bar(bin_edges[:-1], pixel_distribution, align='edge', width=np.diff(bin_edges))
        ax.set_xscale('log')
        ax.set_title(f"Pixel ({idx}, {idx}) Distribution")
        ax.set_ylabel("Probability")
        ax.set_xlabel("Precipitation Value")


    plt.suptitle(title)
    plt.savefig(f'{save_path_name}.png')
    plt.tight_layout()
    plt.show()
    plt.close()


# def plot_distributions(target: torch.Tensor, pred_binned: torch.Tensor,
#                        linspace_binning_inv_norm: np.ndarray, linspace_binning_max_inv_norm: np.ndarray,
#                        title, num_bar_plots=10):
#     """
#     Plot distributions for N pixels sampled uniformly from the diagonal of the picture.
#
#     Args:
#     - target (np.ndarray): Target image, shape (b, 32, 32).
#     - pred_one_hot_log_norm (np.ndarray): Predicted image, shape (b, 64, 32, 32).
#     - linspace_binning_inv_norm (np.ndarray): Left sides of bins.
#     - linspace_binning_max_inv_norm (float): Rightmost bin value.
#     - title (str): Title of the plot.
#     - N (int): Number of pixels to sample.
#     """
#     target = target.detach().cpu().numpy()
#     pred_binned = pred_binned.detach().cpu().numpy()
#
#
#     b, _, h, w = pred_binned.shape
#     fig = plt.figure(figsize=(10, num_bar_plots * 2))
#     gs = GridSpec(num_bar_plots, 1, figure=fig)
#
#     # Sampling N pixels from the diagonal
#     indices = np.linspace(0, h - 1, num_bar_plots, dtype=int)
#     for i, idx in enumerate(indices):
#         ax = fig.add_subplot(gs[i, 0])
#         pixel_distribution = pred_binned[0, :, idx, idx]  # Assuming b = 0 for simplicity
#
#         # Constructing bin edges
#         bin_edges = np.append(linspace_binning_inv_norm, linspace_binning_max_inv_norm)
#
#         # Plotting the distribution as a bar plot
#         ax.bar(bin_edges[:-1], pixel_distribution, align='edge', width=np.diff(bin_edges), log=True)
#         ax.set_title(f"Pixel ({idx}, {idx}) Distribution")
#         ax.set_ylabel("Probability")
#         ax.set_xlabel("Precipitation Value")
#
#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.show()




# def plot_distributions(target, pred_binned, linspace_binning_inv_norm, linspace_binning_max_inv_norm, title, **__):
#     '''
#     Input:
#     pred_one_hot_log_norm: b x c x h x w
#     target: b x w x h
#     '''
#     x=1
#     pass

