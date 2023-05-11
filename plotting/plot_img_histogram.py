import numpy as np
import torch
import matplotlib.pyplot as plt

import gc
# import matplotlib
# matplotlib.use('agg')



# def plot_img_histogram(img, save_path):
#     flat_img = torch.flatten(img)
#     flat_img = flat_img.detach().numpy()
#     counts, bins = np.histogram(flat_img)
#     logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
#     plt.stairs(counts, bins, fill=True)
#     plt.yscale('log')
#     plt.show()



def plot_img_histogram(img, save_path_name, xmin, xmax, s_num_bins_crossentropy, ignore_min_max=False, title='', **_):
    plt.figure()
    flat_img = torch.flatten(img)
    flat_img = flat_img.detach().cpu().numpy()
    plt.subplot(111)
    plt.title(title)
    if not ignore_min_max:
        hist, bins, _ = plt.hist(flat_img, bins=s_num_bins_crossentropy, range=(xmin, xmax))
    else:
        hist, bins, _ = plt.hist(flat_img, bins=s_num_bins_crossentropy)

    plt.xlabel('Log(x+1) normal data')
    plt.yscale('log')

    # histogram on log scale.
    # Use non-equal bin sizes, such that they look equal on log scale.
    # TODO: This does not work ... should work for both pred and input sequence ... however only works for one of them
    # TODO: when either log + 1 or log + 0 FIX THIS BS!!
    # logbins = np.logspace(np.log10(bins[0]++1e-9), np.log10(bins[-1]+1e-9), len(bins))
    # plt.subplot(212)
    # plt.hist(flat_img, bins=logbins)
    # plt.xscale('log')

    plt.savefig(save_path_name, dpi=100)
    plt.show(block=False)

    plt.close("all")
    plt.close()
    gc.collect()
