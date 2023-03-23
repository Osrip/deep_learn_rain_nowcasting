import h5py
import xarray as xr
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from helper_functions import bin_to_one_hot_index_linear
import datetime
from exceptions import CountException
from torch.utils.data import Dataset, DataLoader
import einops
from tqdm import tqdm
# Remember to install package netCDF4 !!


class PrecipitationDataset(Dataset):
    def __init__(self, data_sequence, num_pictures_loaded, num_c_output, linspace_binning_min, linspace_binning_max,
                 normalize=True):
        """
        Attributes:
        self.data_sequence --> log normalized data sequence
        self.data_sequence_one_hot --> log data sequence in one hot encoding
        self.num_pictures_loaded --> number of pictures loded

        returns (tuple in subsequent order):
        log normalized data sequence --> training data frames,
        log one hot data sequence --> target data frame,
        unnormalized data sequence --> training frames
        unnormalized data sequence --> target frame
        """
        self.data_sequence = data_sequence
        self.num_pictures_loaded = num_pictures_loaded

        data_sequence_one_hot, linspace_binning = img_one_hot(data_sequence, num_c_output, linspace_binning_min, linspace_binning_max)
        self.data_sequence_one_hot = einops.rearrange(data_sequence_one_hot, 'i w h c -> i c w h')
        self.linspace_binning = linspace_binning
        # log transform
        # Errors encountered!



    def __len__(self):
        return np.shape(self.data_sequence)[0] - self.num_pictures_loaded

    def __getitem__(self, idx):
        # Returns the first pictures as input data and the last picture as training picture
        return self.data_sequence[idx:idx+self.num_pictures_loaded-1, :, :], \
            self.data_sequence_one_hot[idx+self.num_pictures_loaded, :, :, :], \
            self.data_sequence[idx + self.num_pictures_loaded, :, :], \
            # self.data_sequence_not_normalized[idx:idx+self.num_pictures_loaded-1, :, :], \
            # self.data_sequence_not_normalized[idx:idx + self.num_pictures_loaded, :, :]
            #  [:idx+self.num_pictures_loaded-1] <-- For data_sequence training data (several frames)
            #  [idx+self.num_pictures_loaded, :, :, :] <-- for one_hot target (one frame)


def normalize_data(data_sequence):
    flattened_data = data_sequence.flatten()
    std_data = np.std(flattened_data)
    mean_data = np.mean(flattened_data)
    return (data_sequence - mean_data) / std_data, mean_data, std_data


def inverse_normalize_data(data_sequence, mean_orig_data, std_orig_data):
    return data_sequence * std_orig_data + mean_orig_data


def import_data(input_path, data_keys='/origin1/grid1/category1/entity1/data1/data_matrix1/data',
                flag_keys='/origin1/grid1/category1/entity1/data1/flag_matrix1/flag'):
    hf = h5py.File(input_path)
    data_dataset = hf.get(data_keys)
    flag_dataset = hf.get(flag_keys)
    return data_dataset, flag_dataset


def iterate_through_data_names(start_date_time, future_iterations_from_start: int, minutes_per_iteration: int):
    '''
    start_date_time_ datetime object
    
    Datetime object is initialized with:
    
    datetime.datetime(year, month, day, hour, minute, second, microsecond)
    b = datetime(2022, 12, 28, 23, 55, 59, 342380)
    '''

    load_dates = []
    if minutes_per_iteration % 5 != 0:
        raise CountException('Only 5 minute steps available')

    for i in range(future_iterations_from_start):
        time_diff = minutes_per_iteration * i
        date_time = start_date_time + datetime.timedelta(minutes=time_diff)
        load_dates.append(date_time)
    return load_dates


def flag_data(data_dataset, flag_dataset, nan_letter=0):
    data_arr = np.array(data_dataset)
    flag_arr = np.array(flag_dataset)
    # set all flag values of 0 (data available) True
    booler = lambda x: x == 0
    booler_func = np.vectorize(booler)
    flag_bool = booler_func(flag_arr)
    # Replace all False in flag_bool with nan
    # data_arr[~flag_bool] = np.NAN
    data_arr[~flag_bool] = nan_letter
    return data_arr


def plot_data(data_arr, flag_naming=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    pixel_plot = plt.imshow(data_arr, cmap='Greens', interpolation='nearest', origin='lower')
    plt.colorbar(pixel_plot)
    if flag_naming:
        plt.savefig('misc/flag.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('misc/data.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_data_log(data_arr):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # pixel_plot = plt.imshow(data_arr, cmap='Greens', interpolation='nearest', origin='lower')
    pixel_plot = plt.matshow(data_arr, cmap='Greens', norm=LogNorm(vmin=0.01, vmax=1), interpolation='nearest',
                             origin='lower')
    plt.colorbar(pixel_plot)
    plt.savefig('misc/log_data.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_data_boo(data_arr):
    boo_data = data_arr < 0.001
    pixel_plot = plt.plot(boo_data)
    # plt.colorbar(pixel_plot)
    plt.savefig('misc/data_close_to_zero.png', dpi=300, bbox_inches='tight')
    plt.show()


def img_one_hot(data_arr: np.ndarray, num_c: int, linspace_binning_min, linspace_binning_max):
    '''
    Adds one hot encoded channel dimension
    Channel dimension is added as -1st dimension, so rearrange dimensions!
    '''
    # vmap_mm_to_one_hot_index = np.vectorize(map_mm_to_one_hot_index)
    # data_arr_indexed = vmap_mm_to_one_hot_index(mm=data_arr, max_index=num_c-1, mm_min=mm_min, mm_max=mm_max)
    data_arr_indexed, linspace_binning = bin_to_one_hot_index_linear(data_arr, num_c, linspace_binning_min, linspace_binning_max)
    # data_arr_indexed = bin_to_one_hot_index_log(data_arr, num_c)
    data_indexed = torch.from_numpy(data_arr_indexed)
    data_hot = F.one_hot(data_indexed.long(), num_c)

    return data_hot, linspace_binning


def load_data_sequence_preliminary(folder_path, data_file_name, width_height, data_variable_name, choose_time_span, time_span,
                                   local_machine_mode, **__):
    # TODO: Continue here!

    load_path = '{}/{}'.format(folder_path, data_file_name)
    print('Loading training/validation data from {}'.format(load_path))
    data_dataset = xr.open_dataset(load_path)
    if choose_time_span:
        data_dataset = data_dataset.sel(time=slice(time_span[0], time_span[1]))
    data_arr = data_dataset[data_variable_name].values
    # Get rid of steps dimension
    if local_machine_mode:
        data_arr = data_arr[:, 0, :, :]
    else:
        data_arr = data_arr[0, :, :, :]
    data_tensor = torch.from_numpy(data_arr)

    # Crop --> TODO: Implement this with x y variables of NetCDF in future!
    data_tensor = T.CenterCrop(size=width_height)(data_tensor)
    return data_tensor.numpy()


if __name__ == '__main__':
    input_folder = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/weather_data/dwd_datensatz_bits/rv_recalc/RV_RECALC/hdf/'
    input_file = 'DE1200_RV_Recalc_20201201_0000_+000000.hdf'
    input_path = input_folder + input_file
    data_dataset, flag_dataset = import_data(input_path)
    data_arr = flag_data(data_dataset, flag_dataset)
    plot_data(data_arr)
    plot_data(np.array(flag_dataset), flag_naming=True)
    plot_data_log(data_arr)
    plot_data_boo(data_arr)
    start_date_time = datetime.datetime(2020, 12, 20)
    # iterate_through_data_names(start_date, future_iterations_from_start=3, minutes_per_iteration=5)
    data_sequence = load_data_sequence(start_date_time, input_folder, future_iterations_from_start=3,
                                       minutes_per_iteration=5, width_height=256)

    blub = img_one_hot(data_arr, 64)
    pass


