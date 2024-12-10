import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from helper.helper_functions import convert_tensor_to_np
import torch
import gc
import torchvision.transforms as T
from helper.missing_nan_operations_torch import nanmax

# import matplotlib
# matplotlib.use('agg')


def plot_image(image, save_path_name, vmin, vmax, title=''):
    plt.figure()
    plt.title(title)
    image = np.array(image.cpu())
    im = plt.imshow(image, vmin=vmin, vmax=vmax, norm='linear')
    plt.colorbar(im)
    plt.savefig(save_path_name, dpi=200)
    plt.show(block=False)

    plt.close("all")
    plt.close()
    gc.collect()


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
            curr_ax.imshow(target_img[row, :, :] if col == 0 else pred_img[row, :, :], vmin=vmin, vmax=vmax,
                           norm='linear')
            if row == 0:
                if col == 0:
                    curr_ax.set_title('Targets')
                elif col == 1:
                    curr_ax.set_title('Predictions')
    # plt.colorbar(fig)
    fig.suptitle(title)
    plt.savefig(save_path_name, dpi=600)
    plt.show(block=False)

    plt.close("all")
    plt.close()
    gc.collect()


def plot_target_vs_pred_with_likelihood(target_img, pred_mm, pred_binned, pred_mm_baseline, save_path_name, vmin, vmax, linspace_binning,
                                        plot_baseline,
                                        ps_inv_normalize, max_row_num=5, input_sequence=None, crop_inputs=True,
                                        plot_argmax_probs=True, title='', **__):
    '''
    !!! Can also be plotted without input sequence by just leaving input_sequence=None !!!
    '''
    add_input_sequence = False if input_sequence is None else True

    target_img = convert_tensor_to_np(target_img)
    # pred_mm = convert_tensor_to_np(pred_mm)

    # input_sequence = T.CenterCrop(size=32)(input_sequence)
    if crop_inputs:
        input_sequence = T.CenterCrop(size=32)(input_sequence)
    if add_input_sequence:
        _, _, h, w = input_sequence.shape
        input_sequence = convert_tensor_to_np(input_sequence)
        center_h = h // 2
        center_w = w // 2

        # Calculate the top-left coordinates for the red rectangle
        width_height_target = pred_mm.shape[-1]

        rect_x = center_w - (width_height_target // 2)
        rect_y = center_h - (width_height_target // 2)



    num_rows = np.min((target_img.shape[0], max_row_num))
    add_cols = 0 if input_sequence is None else input_sequence.shape[1]
    # if plot_baseline:
    #     add_baseline = 1
    # else:
    #     add_baseline = 0
    num_cols = 3 + add_cols + plot_baseline + plot_argmax_probs
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5*num_cols, 5*num_rows))
    likelihoods = calc_likelihood_target_vs_pred_man(target_img, pred_binned, linspace_binning)
    # likelihoods = np.log(likelihoods)
    plt.set_cmap('jet')

    for row in range(num_rows):
        for col in range(num_cols):
            curr_ax = axs[row, col]
            if add_input_sequence and col in range(add_cols):
                curr_ax.imshow(input_sequence[row, col, :, :], vmin=vmin, vmax=vmax, norm='linear')

                rect = patches.Rectangle((rect_x, rect_y), width_height_target, width_height_target, linewidth=3,
                                         edgecolor='r', facecolor='none')
                curr_ax.add_patch(rect)
            elif col == 0 + add_cols:
                im1 = curr_ax.imshow(target_img[row, :, :], vmin=vmin, vmax=vmax, norm='linear')
                cbar1 = plt.colorbar(im1)
            elif col == 1 + add_cols:
                # Make this work for both train.py and calc_from_checkpoint
                try:
                    im2 = curr_ax.imshow(pred_mm[row, :, :], vmin=vmin, vmax=vmax, norm='linear')
                except TypeError:
                    # Case of train.py, needs
                    im2 = curr_ax.imshow(pred_mm[row, :, :].detach().cpu(), vmin=vmin, vmax=vmax, norm='linear')

                cbar2 = plt.colorbar(im2, cmap='jet')

            elif col == 2 + add_cols:
                # likelihoods = -np.log(likelihoods)

                im3 = curr_ax.imshow(likelihoods[row, :, :], norm='linear')
                cbar3 = plt.colorbar(im3, cmap='jet')

                if ps_inv_normalize:
                    cbar_label = 'Precipitation forecast in (log?) mm, max 4xSTD'
                else:
                    cbar_label = 'Lognormalized data, max 4xSTD'

                cbar1.set_label(cbar_label, rotation=270, labelpad=12)
                cbar2.set_label(cbar_label, rotation=270, labelpad=12)
                # curr_ax.imshow(target_img[row, :, :] if col == 0 pred_mm[row, :, :], vmin=vmin, vmax=vmax, norm='linear')

            elif (col == 3 + add_cols) and plot_baseline:
                im4 = curr_ax.imshow(pred_mm_baseline[row, :, :], vmin=vmin, vmax=vmax, norm='linear')
                cbar4 = plt.colorbar(im4, cmap='jet')

                cbar4.set_label(cbar_label, rotation=270, labelpad=12)

            elif (col == 4 + add_cols) and plot_argmax_probs:
                pred_binned_max, _ = nanmax(pred_binned, dim=1)

                if torch.max(pred_binned_max[row, :, :]).cpu().numpy() > 1:
                    raise ValueError('Max of binned output should not be larger than 1')

                im5 = curr_ax.imshow(pred_binned_max[row, :, :].cpu().numpy(),
                                     vmin=0,
                                     vmax=1,
                                     norm='linear')
                cbar5 = plt.colorbar(im5, cmap='jet')

                cbar5.set_label('Certainty', rotation=270, labelpad=12)

            if row == 0:
                if add_input_sequence and col in range(add_cols):
                    curr_ax.set_title('Input picture {}'.format(col))
                if col == 0 + add_cols:
                    curr_ax.set_title('Targets')
                elif col == 1 + add_cols:
                    curr_ax.set_title('Predictions')
                elif col == 2 + add_cols:
                    curr_ax.set_title('Likelihood')
                elif (col == 3 + add_cols) and plot_baseline:
                    curr_ax.set_title('Baseline')
                elif (col == 4 + add_cols) and plot_argmax_probs:
                    curr_ax.set_title('Certainty\nMax Probabilities')

    # plt.colorbar(fig)
    fig.suptitle(title)
    plt.savefig('{}.png'.format(save_path_name), dpi=300)
    plt.savefig('{}_bad_qual.png'.format(save_path_name), dpi=50)
    plt.show(block=False)

    plt.close("all")
    plt.close()
    gc.collect()


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