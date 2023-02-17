import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from helper_functions import map_mm_to_one_hot_index
import datetime
from exceptions import CountException
from torch.utils.data import Dataset, DataLoader
import einops
# Remember to install package netCDF4 !!


class PrecipitationDataset(Dataset):
    def __init__(self, data_sequence, num_pictures_loaded, num_c_output, log_transform=True, normalize=True):
        self.data_sequence = data_sequence
        self.num_pictures_loaded = num_pictures_loaded

        mm_min = np.floor(np.min(data_sequence))
        mm_max = np.ceil(np.max(data_sequence))

        # TODO: implement log conversion in one hot
        data_sequence_one_hot = img_one_hot(data_sequence, num_c_output, mm_min, mm_max)
        self.data_sequence_one_hot = einops.rearrange(data_sequence_one_hot, 'i w h c -> i c w h')

        # log transform
        # Errors encountered!
        # Log transform with log x+1 to handle zeros
        # data_sequence = np.log(data_sequence+1)
        # self.data_sequence = normalize_data(data_sequence)

    def __len__(self):
        return np.shape(self.data_sequence)[0] - self.num_pictures_loaded

    def __getitem__(self, idx):
        # Returns the first pictures as input data and the last picture as training picture
        return self.data_sequence[idx:idx+self.num_pictures_loaded-1, :, :], \
            self.data_sequence_one_hot[idx+self.num_pictures_loaded, :, :, :]


def normalize_data(data_sequence):
    flattened_data = data_sequence.flatten()
    std_data = np.std(flattened_data)
    mean_data = np.mean(flattened_data)
    return (data_sequence - mean_data) / std_data


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


def img_one_hot(data_arr: np.ndarray, num_c: int, mm_min, mm_max):
    '''
    Adds one hot encoded channel dimension
    '''
    # TODO: Is it a good idea to add an extra index for nans? ...
    #  probably not a good idea to let the network predict them... but how should we handle them?
    vmap_mm_to_one_hot_index = np.vectorize(map_mm_to_one_hot_index)
    # TODO:pass mm:min mm_max!
    data_arr_indexed = vmap_mm_to_one_hot_index(mm=data_arr, max_index=num_c-2, mm_min=mm_min, mm_max=mm_max)
    data_indexed = torch.from_numpy(data_arr_indexed)
    # TODO: This should be data_arr_indexed!!
    # data_hot = F.one_hot(data_indexed, num_c)
    # Why does this work with long tensor? but not int or float?? Long tensor is Int64!!
    data_hot = F.one_hot(data_indexed.long(), num_c)
    # data_hot = einops.rearrange(data_hot, 'w h c -> c w h')
    # TODO! Seems to work but check, write test!!!
    # Quick check: Compare data_sequence[1,5,5] to one_hot_data_sequence[1,:,5,5]
    return data_hot


def load_data_sequence(start_date_time: datetime.datetime, folder_path: str, future_iterations_from_start: int,
                       minutes_per_iteration: int, width_height: int):
    load_dates = iterate_through_data_names(start_date_time, future_iterations_from_start, minutes_per_iteration)
    data_arr_list = []
    data_sequence = np.empty([len(load_dates), width_height, width_height])
    for i, date in enumerate(load_dates):
        # DE1200_RV_Recalc_20201220_1150_+000000
        # Load Picture
        file_name = 'DE1200_RV_Recalc_{}_{}_+000000.hdf'.format(date.strftime('%Y%m%d'), date.strftime('%H%M'))
        input_path = '{}{}'.format(folder_path, file_name)
        data_dataset, flag_dataset = import_data(input_path)
        data_arr = flag_data(data_dataset, flag_dataset)
        data_tensor = torch.from_numpy(data_arr)
        # Crop Picture
        data_tensor = T.CenterCrop(size=width_height)(data_tensor)
        data_sequence[i, :, :] = data_tensor.numpy()
    return data_sequence


if __name__ == '__main__':
    input_folder = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/weather_data/dwd_datensatz_bits/rv_recalc/RV_RECALC/hdf/'
    input_file = 'DE1200_RV_Recalc_20201201_0000_+000000.hdf'
    input_path = input_folder + input_file
    data_dataset, flag_dataset = import_data(input_path)
    data_arr = flag_data(data_dataset, flag_dataset)
    # plot_data(data_arr)
    # plot_data(np.array(flag_dataset))
    # plot_data_log(data_arr)
    start_date_time = datetime.datetime(2020, 12, 20)
    # iterate_through_data_names(start_date, future_iterations_from_start=3, minutes_per_iteration=5)
    data_sequence = load_data_sequence(start_date_time, input_folder, future_iterations_from_start=3,
                                       minutes_per_iteration=5, width_height=256)

    blub = img_one_hot(data_arr, 64)
    pass


