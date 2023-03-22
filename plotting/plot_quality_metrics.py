import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_relative_mse(relative_mses, save_path_name, title=''):
    plt.figure()
    mean_relative_mses = np.mean(relative_mses, axis=-1)
    plt.plot(mean_relative_mses, label='Training Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('relative MSE')
    plt.savefig(save_path_name, dpi=200)
    plt.show()