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