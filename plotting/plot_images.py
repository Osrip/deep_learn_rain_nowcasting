import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_image(image, save_path_name, vmin, vmax, title=''):
    plt.figure()
    plt.title(title)
    image = np.array(image.cpu())
    im = plt.imshow(image, vmin=vmin, vmax=vmax, norm='linear')
    plt.colorbar(im)
    plt.savefig(save_path_name, dpi=200)
    plt.show()


def plot_target_vs_pred(target_img, pred_img, save_path_name, vmin, vmax, max_row_num=5, title=''):
    convert_tensor_to_np = lambda x: x.cpu().detach().numpy()
    target_img = convert_tensor_to_np(target_img)
    pred_img = convert_tensor_to_np(pred_img)
    num_rows = np.min((target_img.shape[0], max_row_num))
    num_cols = 2
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5*num_rows, 5*num_cols))

    for row in range(num_rows):
        for col in range(num_cols):
            curr_ax = axs[row, col]
            curr_ax.imshow(target_img[row, :, :] if col == 0 else pred_img[row, :, :], vmin=vmin, vmax=vmax, norm='linear')
            if row == 0:
                if col == 0:
                    curr_ax.set_title('Targets')
                elif col == 1:
                    curr_ax.set_title('Predictions')
    # plt.colorbar(fig)
    fig.suptitle(title)
    plt.savefig(save_path_name, dpi=600)
    plt.show()


def plot_target_vs_pred_with_likelihood(target_img, pred_img, save_path_name, vmin, vmax, linspace_binning,
                                        max_row_num=5, title=''):
    convert_tensor_to_np = lambda x: x.cpu().detach().numpy()
    target_img = convert_tensor_to_np(target_img)
    pred_img = convert_tensor_to_np(pred_img)
    num_rows = np.min((target_img.shape[0], max_row_num))
    num_cols = 3
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5*num_rows, 5*num_cols))

    for row in range(num_rows):
        for col in range(num_cols):
            curr_ax = axs[row, col]
            if col == 0:
                curr_ax.imshow(target_img[row, :, :], vmin = vmin, vmax = vmax, norm = 'linear')
            elif col == 1:
                curr_ax.imshow(pred_img[row, :, :], vmin = vmin, vmax = vmax, norm = 'linear')


                # curr_ax.imshow(target_img[row, :, :] if col == 0 pred_img[row, :, :], vmin=vmin, vmax=vmax, norm='linear')
            if row == 0:
                if col == 0:
                    curr_ax.set_title('Targets')
                elif col == 1:
                    curr_ax.set_title('Predictions')
                elif col == 2:
                    curr_ax.set_title('Log-likelihood')
    # plt.colorbar(fig)
    fig.suptitle(title)
    plt.savefig(save_path_name, dpi=600)
    plt.show()



# def plot_image_log(image, save_path_name, vmin, vmax, title=''):
#     fig = plt.figure(figsize=(10, 10))
#     plt.title(title)
#     image = np.array(image.cpu())
#     im = plt.imshow(image, vmin=vmin, vmax=vmax)
#     plt.colorbar(im)
#     plt.savefig(save_path_name, dpi=200)
#     plt.show()


def calc_likelihood_target_vs_pred(target, pred, linspace_binning):
    target = target.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    _get_index_in_linspace_binning_f = lambda x: _get_index_in_linspace_binning(x, linspace_binning)
    _get_index_in_linspace_binning_f = np.vectorize(_get_index_in_linspace_binning_f)
    indecies = _get_index_in_linspace_binning_f(target)
    likelihoods = pred[:, :, :, indecies]


    pass


def _get_index_in_linspace_binning(val, linspace_binning):
    '''
    Takes in value and returns the corresponding index of linspace_binning. Assumes a left-bounded binning
    (value of linspace binning corresponds to lowest value of bin)
    '''
    return np.searchsorted(linspace_binning, val) - 1

