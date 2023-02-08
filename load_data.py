import h5py
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
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
    data_arr[~flag_bool] = np.NAN
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
    pixel_plot = plt.matshow(data_arr, cmap='Greens', norm=LogNorm(vmin=0.01, vmax=1), interpolation='nearest', origin='lower')
    plt.colorbar(pixel_plot)
    plt.show()


if __name__ == '__main__':
    input_folder = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/weather_data/dwd_datensatz_bits/rv_recalc/RV_RECALC/hdf/'
    input_file = 'DE1200_RV_Recalc_20201201_0000_+000000.hdf'
    input_path = input_folder + input_file
    data_dataset, flag_dataset = import_data(input_path)
    data_arr = flag_data(data_dataset, flag_dataset)
    plot_data(data_arr)
    plot_data(np.array(flag_dataset))
    plot_data_log(data_arr)
    pass


