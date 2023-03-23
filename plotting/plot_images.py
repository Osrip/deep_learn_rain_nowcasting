import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_image(image, save_path_name, vmin, vmax, title=''):
    plt.figure()
    plt.title(title)
    im = plt.imshow(image, vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.savefig(save_path_name, dpi=200)
    plt.show()

