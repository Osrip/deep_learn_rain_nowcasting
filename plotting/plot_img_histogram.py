import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_img_histogram(img, save_path):
    flat_img = torch.flatten(img)
    flat_img = flat_img.detach().numpy()
    counts, bins = np.histogram(flat_img)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.stairs(counts, bins, fill=True)
    plt.yscale('log')
    plt.show()


def plot_img_histogram(img, save_path_name, title=''):
    plt.figure()
    flat_img = torch.flatten(img)
    flat_img = flat_img.detach().cpu().numpy()
    plt.subplot(211)
    plt.title(title)
    hist, bins, _ = plt.hist(flat_img, bins=8)

    # histogram on log scale.
    # Use non-equal bin sizes, such that they look equal on log scale.
    # TODO: This does not work ... should work for both pred and input sequence ... however only works for one of them
    # TODO: when either log + 1 or log + 0 FIX THIS BS!!
    logbins = np.logspace(np.log10(bins[0]++1e-9), np.log10(bins[-1]+1e-9), len(bins))
    plt.subplot(212)
    plt.hist(flat_img, bins=logbins)
    plt.xscale('log')

    plt.savefig(save_path_name, dpi=100)
    plt.show()
