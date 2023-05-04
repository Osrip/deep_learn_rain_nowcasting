import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from helper_functions import convert_tensor_to_np


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
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5*num_cols, 5*num_rows))
    plt.set_cmap('jet')

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


def plot_target_vs_pred_with_likelihood(target_img, pred_mm, pred_one_hot, save_path_name, vmin, vmax, linspace_binning,
                                        max_row_num=5, title=''):
    target_img = convert_tensor_to_np(target_img)
    pred_mm = convert_tensor_to_np(pred_mm)
    num_rows = np.min((target_img.shape[0], max_row_num))
    num_cols = 3
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5*num_cols, 5*num_rows))
    likelihoods = calc_likelihood_target_vs_pred_man(target_img, pred_one_hot, linspace_binning)
    # likelihoods = np.log(likelihoods)
    plt.set_cmap('jet')

    for row in range(num_rows):
        for col in range(num_cols):
            curr_ax = axs[row, col]
            if col == 0:
                im1 = curr_ax.imshow(target_img[row, :, :], vmin=vmin, vmax=vmax, norm='linear')
                cbar1 = plt.colorbar(im1)
            elif col == 1:
                im2 = curr_ax.imshow(pred_mm[row, :, :], vmin=vmin, vmax=vmax, norm='linear')
                cbar2 = plt.colorbar(im2, cmap='jet')
            elif col == 2:
                # likelihoods = -np.log(likelihoods)

                im3 = curr_ax.imshow(likelihoods[row, :, :], norm='linear')
                cbar3 = plt.colorbar(im3, cmap='jet')
                cbar_label = 'Precipitation forecast in mm'
                cbar1.set_label(cbar_label, rotation=270, labelpad=2)
                cbar2.set_label(cbar_label, rotation=270)



                # curr_ax.imshow(target_img[row, :, :] if col == 0 pred_mm[row, :, :], vmin=vmin, vmax=vmax, norm='linear')
            if row == 0:
                if col == 0:
                    curr_ax.set_title('Targets')
                elif col == 1:
                    curr_ax.set_title('Predictions')
                elif col == 2:
                    curr_ax.set_title('Log Likelihood')
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


def calc_likelihood_target_vs_pred(target, pred_one_hot, linspace_binning):
    '''
    Not yet finished due to indexing problem!
    '''
    # target = target.detach().cpu().numpy()
    # pred = pred.detach().cpu().numpy()
    pred_one_hot = convert_tensor_to_np(pred_one_hot)
    _get_index_in_linspace_binning_f = lambda x: _get_index_in_linspace_binning(x, linspace_binning)
    _get_index_in_linspace_binning_f = np.vectorize(_get_index_in_linspace_binning_f)
    indecies = _get_index_in_linspace_binning_f(target)
    # In numpy when I have an array X with shape (a,b,c,d) and an index array I with shape (a,b,c) that is filled
    # with indexes that refer to the entries in dimension d. Using X and I now want to create an array Y with shape
    # (a,b,c,d). How do I do that?
    likelihoods = pred_one_hot[:, indecies]
    return likelihoods


def _get_index_in_linspace_binning(val, linspace_binning):
    '''
    Takes in value and returns the corresponding index of linspace_binning. Assumes a left-bounded binning
    (value of linspace binning corresponds to lowest value of bin)
    '''
    return np.searchsorted(linspace_binning, val) - 1


def calc_likelihood_target_vs_pred_man(target, pred_one_hot, linspace_binning):
    pred_one_hot = convert_tensor_to_np(pred_one_hot)
    likelihoods = np.zeros(np.shape(target))
    for b_dim in range(np.shape(target)[0]):
        for w_dim in range(np.shape(target)[1]):
            for h_dim in range(np.shape(target)[2]):
                c_dim = _get_index_in_linspace_binning(target[b_dim, w_dim, h_dim], linspace_binning)
                likelihoods[b_dim, w_dim, h_dim] = pred_one_hot[b_dim, c_dim, w_dim, h_dim]
    return likelihoods