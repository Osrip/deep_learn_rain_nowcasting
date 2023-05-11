import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import gc

# import matplotlib
# matplotlib.use('agg')



def interpolate_smooth(x, y, window_size_smooth=4, polynomial_order_smooth=3, smooth=True, interpolate=True):
    '''
    Interpolating and smoothing for line plots
    For smoothing:
    polyorder must be less than window_length
    window_length must be less than or equal to the size of x

    '''
    if smooth:
        y = savgol_filter(y, window_size_smooth, polynomial_order_smooth)  # window size, polynomial order
    if interpolate:
        f_interpolate = interp1d(x, y, kind='cubic')
        x = np.linspace(np.min(x), np.max(x), num=len(x), endpoint=True)
        y = f_interpolate(x)
    return x, y


def plot_mse_light(mses_list, label_list, save_path_name, title=''):
    plt.figure()
    for mses, label in zip(mses_list, label_list):
        mean_mses = np.mean(mses, axis=-1)
        plt.plot(mean_mses, label=label)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_path_name, dpi=200)
    plt.show(block=False)

    plt.close("all")
    plt.close()
    gc.collect()


def plot_mse_heavy(mses_list, label_list, linestyle_list, color_list, save_path_name, title=''):
    plt.figure()
    for mses, label, linestyle, color in zip(mses_list, label_list, linestyle_list, color_list):
        mean_mses = np.mean(mses, axis=-1)
        plt.plot(mean_mses, label=label, linestyle=linestyle, color=color)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_path_name, dpi=200)
    plt.show(block=False)

    plt.close("all")
    plt.close()
    gc.collect()


def plot_losses(losses, validation_losses, save_path_name):
    mean_losses = np.mean(losses, axis=-1)
    mean_validation_losses = np.mean(validation_losses, axis=-1)
    plt.figure()
    plt.plot(mean_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.savefig('{}/{}_loss.png'.format(plot_dir, s_sim_name), dpi=100)
    # plt.show()
    plt.plot(mean_validation_losses, label='Validation Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_path_name, dpi=100)
    plt.show(block=False)

    plt.close("all")
    plt.close()
    gc.collect()


def plot_average_preds(all_pred_mm, all_target_mm, num_training_samples_per_epoch, save_path_name):
    mean_arr_f = lambda x: np.mean(np.array(x), axis=(1, 2))

    pred_mean = mean_arr_f(all_pred_mm)
    target_mean = mean_arr_f(all_target_mm)

    s_num_epochs = int(pred_mean.shape[0] / num_training_samples_per_epoch)
    mean_per_epoch_arr_f = lambda x: np.array(
        [np.mean([np.array(x)[i + c * num_training_samples_per_epoch] for i in range(num_training_samples_per_epoch)])
         for c in range(s_num_epochs)])

    pred_mean_per_epoch = mean_per_epoch_arr_f(all_pred_mm)
    target_mean_per_epoch = mean_per_epoch_arr_f(all_target_mm)
    x_axis_mean_per_epoch = mean_per_epoch_arr_f(np.arange(len(all_pred_mm)))

    fig = plt.figure(figsize=(10, 7))
    ax2 = fig.add_subplot(212)
    plt.bar(np.arange(len(pred_mean)), pred_mean)
    ylim2 = ax2.get_ylim()
    xlim2 = ax2.get_xlim()

    # plt.vlines(num_training_samples, xlim2[0], xlim2[1] - 0.5, colors='grey', linestyles='--', alpha=0.5)
    vline_indecies = [num_training_samples_per_epoch * i for i in range(s_num_epochs)]
    plt.vlines(vline_indecies, xlim2[0], xlim2[1], colors='grey', linestyles='--', alpha=0.5, linewidth=0.5)


    # Plot means
    try:
        x_mean_per_epoch, y_mean_per_epoch = interpolate_smooth(x_axis_mean_per_epoch, pred_mean_per_epoch,
                                                                window_size_smooth=4,
                                                                polynomial_order_smooth=3)
        plt.plot(x_mean_per_epoch, y_mean_per_epoch, color='red', linewidth='3', alpha=0.5)
    except ValueError:
        warnings.warn('Can only plot smoothed means if window_length must be less than or equal to the size of x')


    plt.xlabel('Training sample #')

    plt.title('Predictions')
    plt.yscale('symlog')

    ax1 = fig.add_subplot(211)
    plt.bar(np.arange(len(target_mean)), target_mean)
    ylim1 = ax1.get_ylim()
    xlim1 = ax1.get_xlim()

    vline_indecies = [num_training_samples_per_epoch * i for i in range(s_num_epochs)]

    plt.vlines(vline_indecies, xlim1[0], xlim1[1], colors='grey', linestyles='--', alpha=0.5, linewidth=0.5)

    try:
        x_mean_per_epoch, y_mean_per_epoch = interpolate_smooth(x_axis_mean_per_epoch, target_mean_per_epoch,
                                                                window_size_smooth=4,
                                                                polynomial_order_smooth=3)
        plt.plot(x_mean_per_epoch, y_mean_per_epoch, color='red', linewidth='2', alpha=0.35)
    except ValueError:
        warnings.warn('Can only plot smoothed means if window_length must be less than or equal to the size of x')


    plt.title('Targets')
    plt.yscale('symlog')

    if ylim1[1] > ylim2[1]:
        ax1.set_ylim(ylim1)
        ax2.set_ylim(ylim1)
    else:
        ax1.set_ylim(ylim2)
        ax2.set_ylim(ylim2)

    plt.savefig(save_path_name, dpi=600)
    plt.show(block=False)

    plt.close("all")
    plt.close()
    gc.collect()


def plot_pixelwise_preds(all_pred_mm, all_target_mm, epoch, save_path_name, swap_x_y=True):

    all_pred_mm = np.array(all_pred_mm)
    all_target_mm = np.array(all_target_mm)
    reshape_f = lambda x: x.reshape(x.shape[0], -1)
    reshaped_pred_mm = reshape_f(all_pred_mm)
    reshaped_target_mm = reshape_f(all_target_mm)
    if not swap_x_y:
        reshaped_pred_mm = np.swapaxes(reshaped_pred_mm, 0, 1)
        reshaped_target_mm = np.swapaxes(reshaped_target_mm, 0, 1)

    xlabel = 'Training sample'
    ylabel = 'Pixel #'

    if swap_x_y:
        tmp = ylabel
        ylabel = xlabel
        xlabel = tmp

    plt.figure(figsize=(10, (epoch+1)*3))
    plt.subplot(211)
    plt.imshow(reshaped_target_mm)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Targets')
    plt.subplot(212)
    plt.imshow(reshaped_pred_mm)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Predictions')
    plt.savefig(save_path_name, dpi=300, bbox_inches='tight')
    plt.show(block=False)

    plt.close("all")
    plt.close()
    gc.collect()