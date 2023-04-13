import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_mse(mses_list, label_list, save_path_name, title=''):
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
    plt.show()


def plot_losses(losses, validation_losses, save_path_name):
    mean_losses = np.mean(losses, axis=-1)
    mean_validation_losses = np.mean(validation_losses, axis=-1)
    plt.figure()
    plt.plot(mean_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.savefig('{}/{}_loss.png'.format(plot_dir, sim_name), dpi=100)
    # plt.show()
    plt.plot(mean_validation_losses, label='Validation Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_path_name, dpi=100)
    plt.show()


def plot_average_preds(all_pred_mm, all_target_mm, save_path_name):
    mean_arr_f = lambda x: np.mean(np.array(x), axis=(1,2))

    pred_mean = mean_arr_f(all_pred_mm)
    target_mean = mean_arr_f(all_target_mm)

    plt.figure(figsize=(10, 7))
    plt.subplot(211)
    plt.bar(np.arange(len(target_mean)), target_mean)
    plt.title('Targets')
    plt.subplot(212)
    plt.bar(np.arange(len(pred_mean)), pred_mean)
    plt.title('Predictions')
    plt.show()
    plt.savefig(save_path_name, dpi=100)
    plt.show()


def plot_pixelwise_preds(all_pred_mm, all_target_mm, save_path_name, swap_x_y=True):

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

    plt.figure(figsize=(10, 10))
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
    plt.savefig(save_path_name, dpi=600, bbox_inches='tight')
    plt.show()


