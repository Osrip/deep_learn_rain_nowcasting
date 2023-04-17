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


# def plot_image_log(image, save_path_name, vmin, vmax, title=''):
#     fig = plt.figure(figsize=(10, 10))
#     plt.title(title)
#     image = np.array(image.cpu())
#     im = plt.imshow(image, vmin=vmin, vmax=vmax)
#     plt.colorbar(im)
#     plt.savefig(save_path_name, dpi=200)
#     plt.show()
