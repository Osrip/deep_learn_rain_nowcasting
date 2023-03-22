import numpy as np
import torch
import matplotlib.pyplot as plt





def plot_image(image, save_path_name, title=''):
    plt.figure()
    plt.title(title)
    plt.imshow(image)
    plt.savefig(save_path_name, dpi=200)
    plt.show()

