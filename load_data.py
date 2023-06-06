import h5py
import xarray as xr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from helper.helper_functions import bin_to_one_hot_index_linear, chunk_list, flatten_list
import datetime
from exceptions import CountException
from torch.utils.data import Dataset
import einops


# Remember to install package netCDF4 !!


# class PrecipitationDataset(Dataset):
#     def __init__(self, data_sequence, last_input_rel_idx, target_rel_idx, num_c_output, linspace_binning_min,
#                  linspace_binning_max, normalize=True):
#         """
#         Attributes:
#         self.data_sequence --> log normalized data sequence
#         self.data_sequence_one_hot --> log data sequence in one hot encoding
#         self.num_pictures_loaded --> number of pictures loded
#
#         returns (tuple in subsequent order):
#         log normalized data sequence --> training data frames,
#         log one hot data sequence --> target data frame,
#         unnormalized data sequence --> training frames
#         unnormalized data sequence --> target frame
#         """
#         self.data_sequence = data_sequence
#         self.normalize = normalize
#         self.last_input_rel_idx = last_input_rel_idx
#         self.target_rel_idx = target_rel_idx
#         self.transform_f = transform_f
#
#         data_sequence_one_hot, linspace_binning = img_one_hot(data_sequence, num_c_output, linspace_binning_min, linspace_binning_max)
#         self.data_sequence_one_hot = einops.rearrange(data_sequence_one_hot, 'i w h c -> i c w h')
#         self.linspace_binning = linspace_binning
#         # log transform
#         # Errors encountered!
#
#     def __len__(self):
#         return np.shape(self.data_sequence)[0] - self.target_rel_idx
#
#     def __getitem__(self, idx):
#         # Returns the first pictures as input data and the last picture as training picture
#         return self.data_sequence[idx:idx+self.last_input_rel_idx, :, :], \
#             self.data_sequence_one_hot[idx+self.target_rel_idx, :, :, :], \
#             self.data_sequence[idx + self.target_rel_idx, :, :], \
#             # TODO Make sure Indexing does what it's supposed to do!
#             # Output: Training sequence mm, target one hot, target mm
#
#             # self.data_sequence_not_normalized[idx:idx+self.num_pictures_loaded-1, :, :], \
#             # self.data_sequence_not_normalized[idx:idx + self.num_pictures_loaded, :, :]
#             #  [:idx+self.num_pictures_loaded-1] <-- For data_sequence training data (several frames)
#             #  [idx+self.num_pictures_loaded, :, :, :] <-- for one_hot target (one frame)


class PrecipitationFilteredDataset(Dataset):
    def __init__(self, filtered_data_loader_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min, linspace_binning_max, linspace_binning, transform_f,
                 s_num_bins_crossentropy, s_folder_path, s_width_height, s_width_height_target, s_data_variable_name,
                 s_local_machine_mode, s_normalize=True, **__):
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



        self.linspace_binning = linspace_binning # Gets assigned in __getitem__
        # TODO pretty ugly due to evolutionary code history... Create linspace binning at the beginning in train.py at some point

        self.filtered_data_loader_indecies = filtered_data_loader_indecies

        self.s_folder_path = s_folder_path
        self.transform_f = transform_f
        self.s_width_height = s_width_height
        self.s_width_height_target = s_width_height_target
        self.s_normalize = s_normalize
        self.s_num_bins_crossentropy = s_num_bins_crossentropy
        self.linspace_binning_min = linspace_binning_min
        self.linspace_binning_max = linspace_binning_max
        self.mean_filtered_data = mean_filtered_data
        self.std_filtered_data = std_filtered_data
        self.s_data_variable_name = s_data_variable_name
        self.s_local_machine_mode = s_local_machine_mode

        # log transform
        # Errors encountered!

    def __len__(self):
        return len(self.filtered_data_loader_indecies)

    def __getitem__(self, idx):
        # Loading everything directly from the disc
        # Returns the first pictures as input data and the last picture as training picture
        filtered_data_loader_indecies_dict = self.filtered_data_loader_indecies[idx]
        file = filtered_data_loader_indecies_dict['file']
        first_idx_input_sequence = filtered_data_loader_indecies_dict['first_idx_input_sequence']
        last_idx_input_sequence = filtered_data_loader_indecies_dict['last_idx_input_sequence']
        target_idx_input_sequence = filtered_data_loader_indecies_dict['target_idx_input_sequence']
        data_dataset = xr.open_dataset('{}/{}'.format(self.s_folder_path, file))
        input_data_set = data_dataset.isel(time=slice(first_idx_input_sequence, last_idx_input_sequence)) # last_idx_input_sequence + 1 like in np! Did I already do that prior?

        input_sequence = input_data_set[self.s_data_variable_name].values
        # Get rid of steps dimension
        input_sequence = input_sequence[:, 0, :, :]
        # if self.s_local_machine_mode:
        #     input_sequence = input_sequence[:, 0, :, :]
        # else:
        #     input_sequence = input_sequence[0, :, :, :]

        input_sequence = np.array(T.CenterCrop(size=self.s_width_height)(torch.from_numpy(input_sequence)))
        input_sequence = lognormalize_data(input_sequence, self.mean_filtered_data, self.std_filtered_data, self.transform_f, self.s_normalize)
        target_data_set = data_dataset.isel(time=target_idx_input_sequence)
        target = target_data_set[self.s_data_variable_name].values
        del data_dataset
        # Get rid of steps dimension as we only have one index anyways
        # TODO: Check what this does on Slurm with non-test data!
        target = target[0]
        target = np.array(T.CenterCrop(size=self.s_width_height_target)(torch.from_numpy(target)))
        target = lognormalize_data(target, self.mean_filtered_data, self.std_filtered_data, self.transform_f, self.s_normalize)

        # TODO DEBUGGING REMOVE THIS:

        target_one_hot = img_one_hot(target, self.s_num_bins_crossentropy, self.linspace_binning)

        target_one_hot = einops.rearrange(target_one_hot, 'w h c -> c w h')

        # Float conversion should

        return input_sequence, target_one_hot, target
        # TODO: Returning linspace binning here every time is super ugly as this is a global constant!!!


def lognormalize_data(data, mean_data, std_data, transform_f, s_normalize):
    data = transform_f(data)
    if s_normalize:
        data, _, _ = normalize_data(data, mean_data=mean_data, std_data=std_data)
    return data


def filtering_data_scraper(transform_f, last_input_rel_idx, target_rel_idx, s_folder_path, s_data_file_names, s_width_height, s_data_variable_name,
                           s_time_span, s_local_machine_mode, s_width_height_target, s_min_rain_ratio_target, s_choose_time_span=False, **__):
    '''
    time span only refers to a single file
    '''
    filtered_data_loader_indecies = []

    # num_x and sum_x as well as sum_x_squared are used to calculate the first and second momentum for data normalization
    num_x = 0  # num_x is the number of data points, meaning the TOTAL NUMBER OF PIXELS, that have passed the filter
    sum_x = 0     # sum_x and sum_x squared are the sums of all data points / the squres of all datapoints
    sum_x_squared = 0
    num_frames_passed_filter = 0

    num_frames_total = 0

    linspace_binning_min_unnormalized = np.inf
    linspace_binning_max_unnormalized = -np.inf

    for data_file_name in s_data_file_names:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: DONT ALLOW s_time_span CHOOSING, SCREWS UP INDECIES WHEN LOADING DATA
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        curr_data_sequence = load_data_sequence_preliminary(s_folder_path, data_file_name, s_width_height, s_data_variable_name,
                                                       s_choose_time_span, s_time_span, s_local_machine_mode)
        for i in range(np.shape(curr_data_sequence)[0] - target_rel_idx):
            num_frames_total += 1
            first_idx_input_sequence = i
            last_idx_input_sequence = i + last_input_rel_idx
            target_idx_input_sequence = i + target_rel_idx

            curr_target_cropped = np.array(T.CenterCrop(size=s_width_height_target)(torch.from_numpy(curr_data_sequence[target_idx_input_sequence])))
            curr_input_sequence = curr_data_sequence[first_idx_input_sequence:last_idx_input_sequence, :, :]
            if filter(curr_input_sequence, curr_target_cropped, s_min_rain_ratio_target):
                num_frames_passed_filter += 1
                filtered_data_loader_indecies_dict = {}
                filtered_data_loader_indecies_dict['file'] = data_file_name
                filtered_data_loader_indecies_dict['first_idx_input_sequence'] = first_idx_input_sequence
                filtered_data_loader_indecies_dict['last_idx_input_sequence'] = last_idx_input_sequence
                filtered_data_loader_indecies_dict['target_idx_input_sequence'] = target_idx_input_sequence
                filtered_data_loader_indecies.append(filtered_data_loader_indecies_dict)


                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # ! IGNORES FIRST ENTRIES: For means and std to normalize data only the values of the target sequence are taken !
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                # We are iterating through all 256x256 target frames that have been accepted by the filter

                # !!!TODO: !!! num_x is zero when filter is never applied! Account for this !!!
                num_x += np.shape(curr_data_sequence[target_idx_input_sequence].flatten())[0]
                sum_x += np.sum(transform_f(curr_data_sequence[target_idx_input_sequence].flatten()))
                sum_x_squared += np.sum(transform_f(curr_data_sequence[target_idx_input_sequence].flatten()) ** 2)

                # linspace binning min and max have to be normalized later as the means and stds are available

                # min_curr_input_and_target = np.min(
                #         curr_data_sequence[np.r_[i:last_idx_input_sequence, target_idx_input_sequence]])
                # TODO: previously min was taken from 256x256 target instead of 36x36. Is Bug now fixed?
                min_input = np.min(curr_data_sequence[i:last_idx_input_sequence])
                min_target = np.min(curr_data_sequence[target_idx_input_sequence])
                min_curr_input_and_target = np.min([min_input, min_target])

                if min_curr_input_and_target < 0:
                    # This should never occur as Filter should filter out all negative values
                    raise Exception('Values smaller than 0 within the test and validation dataset. Probably NaNs')

                max_curr_input_and_target = np.max(
                        curr_data_sequence[np.r_[i:last_idx_input_sequence, target_idx_input_sequence]])

                if linspace_binning_min_unnormalized > min_curr_input_and_target:
                    linspace_binning_min_unnormalized = min_curr_input_and_target

                if linspace_binning_max_unnormalized < max_curr_input_and_target:
                    linspace_binning_max_unnormalized = max_curr_input_and_target

            # TODO: Write a test for this!!
            # TODO: Is Bessel's correction (+1 accounting for extra degree of freedom) needed here?
        if num_x == 0:
            raise Exception('No data passed the filter conditions of s_min_rain_ratio_target={}, such that there is no '
                            'data for training and validation.'.format(s_min_rain_ratio_target))
        else:
            print('{} data points out of a total of {} scanned data points'
                  ' passed the filter condition of s_min_rain_ratio_target={}'.format(
                num_frames_passed_filter, num_frames_total, s_min_rain_ratio_target))
        mean_filtered_data = sum_x / num_x
        std_filtered_data = np.sqrt((sum_x_squared / num_x) - mean_filtered_data ** 2)
    return filtered_data_loader_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized,\
        linspace_binning_max_unnormalized


# def filter(input_sequence, target, s_min_rain_ratio_target):
#     '''
#     Looks whether on what percentage on target there is rain. If percentage exceeds s_min_rain_ratio_target return True
#     , False otherwise
#     '''
#     # Todo: Implement filter!
#     # rainy_data_points = target[target > 0]
#     # Only take frames, that have any pixels that are neither 0 nor NaN AND pixels that have no NaNs
#     if ((target != -1000000000.0) & (target != 0)).any() and \
#             not (target == -1000000000.0).any() and \
#             not (input_sequence == -1000000000.0).any():
#         return True
#     else:
#         return False



# def filter(input_sequence, target, s_min_rain_ratio_target, percentage=1):
#     '''
#     Looks whether on what percentage on target there is rain. If percentage exceeds s_min_rain_ratio_target return True
#     , False otherwise
#     percentage 0.3 default
#     '''
#     # Todo: Implement filter!
#     # rainy_data_points = target[target > 0]
#     # Only take frames, that have any pixels that are neither 0 nor NaN AND pixels that have no NaNs
#     if (target[target != 0].size > percentage * target.size) and \
#             not (target == -1000000000.0).any() and \
#             not (input_sequence == -1000000000.0).any():
#         return True
#     else:
#         return False




def filter(input_sequence, target, s_min_rain_ratio_target, percentage=0.5, min_amount_rain=0.05):

    '''
    reasonable amount of data passes: percentage=0.5, min_amount_rain=0.05
    '''

    # I used to check for not (target == -1000000000.0).any() and \ not (input_sequence == -1000000000.0).any() to throw out
    # NaNs. However somewhere in data set there seem to be values that are < 0 and != 10000000000.0, TODO no idea what that is..
    if (target[target > min_amount_rain].size > percentage * target.size) and \
            not (target < 0).any() and \
            not (input_sequence < 0).any():
        return True
    else:
        return False


def normalize_data(data_sequence, mean_data=None, std_data=None):
    flattened_data = data_sequence.flatten()
    if mean_data is None:
        mean_data = np.mean(flattened_data)
    if std_data is None:
        std_data = np.std(flattened_data)
    return (data_sequence - mean_data) / std_data, mean_data, std_data


def inverse_normalize_data(data_sequence, mean_orig_data, std_orig_data, inverse_log=True, inverse_normalize=True):
    '''
    Assumes that the original data has been logtransformed first and subsequently normalized to standard normal
    '''

    if inverse_normalize:
        data_sequence = data_sequence * std_orig_data + mean_orig_data
    if inverse_log:
        data_sequence = np.exp(data_sequence) - 1

    return data_sequence

    # if inverse_log:
    #     inverse_normalized = data_sequence * std_orig_data + mean_orig_data
    #     return np.exp(inverse_normalized) - 1
    # else:
    #     return data_sequence * std_orig_data + mean_orig_data


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


def img_one_hot(data_arr: np.ndarray, num_c: int, linspace_binning):
    '''
    Adds one hot encoded channel dimension
    Channel dimension is added as -1st dimension, so rearrange dimensions!
    '''
    # TODO: Looks like only first third of bins is used in practice despite lognormalization!
    # vmap_mm_to_one_hot_index = np.vectorize(map_mm_to_one_hot_index)
    # data_arr_indexed = vmap_mm_to_one_hot_index(mm=data_arr, max_index=num_c-1, mm_min=mm_min, mm_max=mm_max)
    data_arr_indexed = bin_to_one_hot_index_linear(data_arr, linspace_binning) # -0.00000001
    # data_arr_indexed = bin_to_one_hot_index_log(data_arr, num_c)
    data_indexed = torch.from_numpy(data_arr_indexed)

    data_hot = F.one_hot(data_indexed.long(), num_c)

    return data_hot


def load_data_sequence_preliminary(s_folder_path, data_file_name, s_width_height, s_data_variable_name, s_choose_time_span,
                                   s_time_span, s_local_machine_mode, **__):
    '''
    This function loads one file and has the option to load a subset of the file instead by setting s_choose_time_span to true
    '''
    # TODO: Continue here!

    load_path = '{}/{}'.format(s_folder_path, data_file_name)
    print('Loading training/validation data from {}'.format(load_path))
    data_dataset = xr.open_dataset(load_path)
    if s_choose_time_span:
        # TODO Change this back, only for test purposes!!
        data_dataset = data_dataset.isel(time=slice(s_time_span[0], s_time_span[1]))
        # data_dataset = data_dataset.sel(time=slice(s_time_span[0], s_time_span[1]))
    data_arr = data_dataset[s_data_variable_name].values
    data_arr = data_arr[:, 0, :, :]
    # Get rid of steps dimension
    # if s_local_machine_mode:
    #     data_arr = data_arr[:, 0, :, :]
    # else:
    #     data_arr = data_arr[0, :, :, :]
    data_tensor = torch.from_numpy(data_arr)

    # Crop --> TODO: Implement this with x y variables of NetCDF in future!
    data_tensor = T.CenterCrop(size=s_width_height)(data_tensor)
    return data_tensor.numpy()


def random_splitting_filtered_indecies(indecies, num_training_samples, num_validation_samples, chunk_size):
    chunked_indecies = chunk_list(indecies, chunk_size)
    num_chunks_training = int(num_training_samples / chunk_size)
    num_chunks_validation = len(chunked_indecies) - num_chunks_training
    chunk_idxs = np.arange(len(chunked_indecies))
    np.random.shuffle(chunk_idxs)
    training_chunks = [chunked_indecies[i] for i in chunk_idxs[0:num_chunks_training]]
    vaildation_chunks = [chunked_indecies[i] for i in chunk_idxs[num_chunks_training:]]
    training_indecies = flatten_list(training_chunks)
    validation_indecies = flatten_list(vaildation_chunks)
    return training_indecies, validation_indecies


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
                                       minutes_per_iteration=5, s_width_height=256)

    blub = img_one_hot(data_arr, 64)
    pass


