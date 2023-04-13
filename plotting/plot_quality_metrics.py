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


def plot_pixelwise_preds(all_pred_mm, all_target_mm, save_path_name):
    plt.figure(figsize=(10, 7))
    plt.subplot(211)
    plt.bar(np.arange(len(all_target_mm)), all_target_mm)
    plt.title('Predictions')
    plt.subplot(212)
    plt.bar(np.arange(len(all_pred_mm)), all_pred_mm)
    plt.title('Predictions')
    plt.show()
    plt.savefig(save_path_name, dpi=100)
    plt.show()


