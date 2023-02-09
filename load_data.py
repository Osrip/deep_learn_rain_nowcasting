import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from helper_functions import map_mm_to_one_hot_index
# Remember to install package netCDF4 !!


def import_data(input_path, data_keys='/origin1/grid1/category1/entity1/data1/data_matrix1/data',
                flag_keys='/origin1/grid1/category1/entity1/data1/flag_matrix1/flag'):
    hf = h5py.File(input_path)
    data_dataset = hf.get(data_keys)
    flag_dataset = hf.get(flag_keys)
    return data_dataset, flag_dataset


def flag_data(data_dataset, flag_dataset):
    data_arr = np.array(data_dataset)
    flag_arr = np.array(flag_dataset)
    # set all flag values of 0 (data available) True
    booler = lambda x: x == 0
    booler_func = np.vectorize(booler)
    flag_bool = booler_func(flag_arr)
    # Replace all False in flag_bool with nan
    # data_arr[~flag_bool] = np.NAN
    data_arr[~flag_bool] = 0
    return data_arr


def plot_data(data_arr):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    pixel_plot = plt.imshow(data_arr, cmap='Greens', interpolation='nearest', origin='lower')
    plt.colorbar(pixel_plot)
    plt.show()


def plot_data_log(data_arr):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # pixel_plot = plt.imshow(data_arr, cmap='Greens', interpolation='nearest', origin='lower')
    pixel_plot = plt.matshow(data_arr, cmap='Greens', norm=LogNorm(vmin=0.01, vmax=1), interpolation='nearest',
                             origin='lower')
    plt.colorbar(pixel_plot)
    plt.show()


def img_one_hot(data_arr: np.ndarray, num_c: int):
    '''
    Adds one hot encoded channel dimension
    '''
    # TODO: Is it a good idea to add an extra index for nans? ...
    #  probably not a good idea to let the network predict them... but how should we handle them?
    vmap_mm_to_one_hot_index = np.vectorize(map_mm_to_one_hot_index)
    data_arr_indexed = vmap_mm_to_one_hot_index(mm=data_arr, max_index=64, mm_min=0, mm_max=20)
    data_indexed = torch.from_numpy(data_arr)
    data_hot = F.one_hot(data_indexed, num_c)
    # TODO Why does this work with long tensor? but not int or float?? Long temnsor is Int64!!
    data_hot = F.one_hot(data_indexed.long(), 64)
    # TODO! Seems to work but check, write test!!!
    return data_hot


if __name__ == '__main__':
    input_folder = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/weather_data/dwd_datensatz_bits/rv_recalc/RV_RECALC/hdf/'
    input_file = 'DE1200_RV_Recalc_20201201_0000_+000000.hdf'
    input_path = input_folder + input_file
    data_dataset, flag_dataset = import_data(input_path)
    data_arr = flag_data(data_dataset, flag_dataset)
    # plot_data(data_arr)
    # plot_data(np.array(flag_dataset))
    # plot_data_log(data_arr)

    blub = img_one_hot(data_arr, 64)
    pass


