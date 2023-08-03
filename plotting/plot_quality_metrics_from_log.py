import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import gc
import pandas as pd
# import _set_wdir

# import matplotlib
# matplotlib.use('agg')

def interpolate_smooth(x, y, window_size_smooth=4, polynomial_order_smooth=3, num_data_points_interp=4000, smooth=True, interpolate=True):
    '''
    Interpolating and smoothing for line plots
    For smoothing:
    polyorder must be less than window_length
    window_length must be less than or equal to the size of x
    Assumes linspace of x and accordingly ordered y

    '''
    if smooth:
        y = savgol_filter(y, window_size_smooth, polynomial_order_smooth)  # window size, polynomial order
    if interpolate:
        f_interpolate = interp1d(x, y, kind='cubic')
        x = np.linspace(np.min(x), np.max(x), num=num_data_points_interp, endpoint=True)
        y = f_interpolate(x)
    return x, y


def plot_mse_light(mean_mses, label_list, save_path_name, title=''):
    plt.figure()
    for mean_mses, label in zip(mean_mses, label_list):
        plt.plot(mean_mses, label=label)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_path_name, dpi=200, bbox_inches='tight')
    plt.show(block=False)

    plt.close("all")
    plt.close()
    gc.collect()


def plot_mse_heavy(mean_mses, label_list, linestyle_list, color_list, save_path_name, ylabel='MSE', ylog=True, title=''):
    plt.figure()
    for mses, label, linestyle, color in zip(mean_mses, label_list, linestyle_list, color_list):
        plt.plot(mses, label=label, linestyle=linestyle, color=color)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
    if ylog:
        plt.yscale('log')
    plt.legend()
    plt.savefig(save_path_name, dpi=200, bbox_inches='tight')
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
    plt.savefig(save_path_name, dpi=100, bbox_inches='tight')
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

    plt.savefig(save_path_name, dpi=600, bbox_inches='tight')
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

def load_data(ps_sim_name, **__):
    rel_path_train = 'logs/train_log/version_0/metrics.csv'
    rel_path_val = 'logs/val_log/version_0/metrics.csv'

    train_df = pd.read_csv('{}/{}'.format(ps_sim_name, rel_path_train))
    val_df = pd.read_csv('{}/{}'.format(ps_sim_name, rel_path_val))

    return train_df, val_df


def df_cols_to_list_of_lists(keys, df):
    out_list = []
    for key in keys:
        out_list.append(df[key].to_list())
    return out_list


def plot_mse_manual(train_df, val_df, ps_sim_name, **__):
    key_list_train = ['train_mse_pred_target', 'train_mse_zeros_target',
                      'train_mse_persistence_target']
    train_mse_list = df_cols_to_list_of_lists(key_list_train, train_df)

    key_list_val = ['val_mse_pred_target', 'val_mse_zeros_target',
                    'val_mse_persistence_target']

    val_mse_list = df_cols_to_list_of_lists(key_list_val, val_df)

    mse_list = train_mse_list + val_mse_list
    key_list = key_list_train + key_list_val

    plot_mse_heavy(mean_mses=mse_list,
                   label_list=key_list,
                   color_list=['g', 'y', 'b', 'g', 'y', 'b'], linestyle_list=['-', '-', '-', '--', '--', '--'],
                   save_path_name='{}/plots/mse_with_val'.format(ps_sim_name),
                   title='MSE on lognorm data')


def line_plot(train_df, val_df, key_list_train, key_list_val, save_name, ps_sim_name, ylog=True, ylabel='MSE',
              title='', color_list=[None for i in range(99)],
              linestyle_list=[None for i in range(99)], **__):

    train_mse_list = df_cols_to_list_of_lists(key_list_train, train_df)
    val_mse_list = df_cols_to_list_of_lists(key_list_val, val_df)
    mse_list = train_mse_list+val_mse_list
    key_list = key_list_train + key_list_val

    plot_mse_heavy(mean_mses=mse_list,
                   label_list=key_list,
                   color_list=color_list, linestyle_list=linestyle_list,
                   save_path_name='{}/plots/{}'.format(ps_sim_name, save_name),
                   ylabel=ylabel,
                   ylog=ylog,
                   title=title)


def plot_qualities_main(plot_settings, ps_sim_name, s_gaussian_smoothing_target, **__):
    '''
    Plots both mse vals and loss. Adjust loss according to what is calculated
    (xentropy or kl divergence dependong on whether s_gaussian_smoothing_target is active)
    This requires both **plot_settings and **settings as input
    '''
    train_df, val_df = load_data(**plot_settings)
    key_list_train_mse = ['train_mse_pred_target', 'train_mse_zeros_target',
                      'train_mse_persistence_target']
    key_list_val_mse = ['val_mse_pred_target', 'val_mse_zeros_target',
                    'val_mse_persistence_target']

    line_plot(train_df, val_df, key_list_train_mse, key_list_val_mse, save_name='mse_with_val',
              color_list=['g', 'y', 'b', 'g', 'y', 'b'], linestyle_list=['-', '-', '-', '--', '--', '--'],
              title='MSE on lognorm data', **plot_settings,)

    key_list_train_xentropy = ['train_loss']
    key_list_val_xentropy = ['val_loss']

    # if s_gaussian_smoothing_target:
    #     loss_ylog = False
    #     loss_ylabel = 'KL Divergence'
    #     loss_title = 'KL divergence on lognorm data'
    # else:
    loss_ylog = True
    loss_title = 'Xentropy on lognorm data'
    loss_ylabel = 'Xentropy'

    line_plot(train_df, val_df, key_list_train_xentropy, key_list_val_xentropy, ylabel=loss_ylabel, ylog=loss_ylog,
              save_name='xentropy_loss', title=loss_title, **plot_settings)


def plot_precipitation_diff(plot_settings, ps_sim_name, **__):
    '''
    Plots both mse vals and loss. Adjust loss according to what is calculated
    (xentropy or kl divergence dependong on whether s_gaussian_smoothing_target is active)
    This requires both **plot_settings and **settings as input
    '''
    train_df, val_df = load_data(**plot_settings)
    key_list_train_diff = ['train_mean_diff_pred_target_mm']
    key_list_val_diff = ['val_mean_diff_pred_target_mm']

    line_plot(train_df, val_df, key_list_train_diff, key_list_val_diff, save_name='mean_diff_mm',
              color_list=['b', 'y'], linestyle_list=['-', '--'],
              title='Mean Precipitation difference prediciton - target', ylabel='Mean Difference (lognormalized data)', ylog=False,
              **plot_settings,)

    key_list_train_mm = ['train_mean_pred_mm', 'train_mean_target_mm']
    key_list_val_mm = ['val_mean_pred_mm', 'val_mean_target_mm']

    # TODO: Mark target and prediction!!!
    line_plot(train_df, val_df, key_list_train_mm, key_list_val_mm, save_name='mean_mm',
              color_list=['b', 'y', 'b', 'y'], linestyle_list=['-', '-', '--', '--'],
              title='Total precipitation of Prediciton and target', ylabel='Mean Precipitation [mm]', ylog=False,
              **plot_settings,)


if __name__ == '__main__':
    plot_settings = {
        'ps_sim_name': None # Insert folder to run here,
    }
    plot_qualities_main(plot_settings, s_gaussian_smoothing_target=False, **plot_settings)
