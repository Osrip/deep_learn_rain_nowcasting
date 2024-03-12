import h5py
import xarray as xr
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from helper.helper_functions import chunk_list, flatten_list, img_one_hot
import datetime
from exceptions import CountException
from torch.utils.data import Dataset
import einops
import random


# Remember to install package netCDF4 !!

class PrecipitationFilteredDataset(Dataset):
    def __init__(self, filtered_data_loader_indecies, mean_filtered_log_data, std_filtered_log_data, linspace_binning_min, linspace_binning_max, linspace_binning, transform_f,
                 s_num_bins_crossentropy, s_folder_path, s_width_height, s_width_height_target, s_data_variable_name,
                 s_local_machine_mode, device, s_normalize=True, **__):
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
        self.mean_filtered_log_data = mean_filtered_log_data
        self.std_filtered_log_data = std_filtered_log_data
        self.s_data_variable_name = s_data_variable_name
        self.s_local_machine_mode = s_local_machine_mode
        self.s_device = device
        # log transform
        # Errors encountered!

    def __len__(self):
        return len(self.filtered_data_loader_indecies)

    def __getitem__(self, idx):
        # Loading everything directly from the disc
        # Returns the first pictures as input data and the last picture as training picture
        input_sequence, target_one_hot, target, target_one_hot_extended = \
            load_input_target_from_index(idx, self.filtered_data_loader_indecies, self.linspace_binning,
                                     self.mean_filtered_log_data, self.std_filtered_log_data, self.transform_f,
                                     self.s_width_height, self.s_width_height_target, self.s_data_variable_name,
                                     self.s_normalize, self.s_num_bins_crossentropy, self.s_folder_path, self.s_device,
                                     normalize=True, load_input_sequence=True, load_target=True
                                     )

        return input_sequence, target_one_hot, target, target_one_hot_extended


# TODO: !!!! rewrite this such that it only loads extended target if we are really doing gausian smoothing! !!!
def load_input_target_from_index(idx, filtered_data_loader_indecies, linspace_binning, mean_filtered_log_data, std_filtered_log_data,
                                 transform_f, s_width_height, s_width_height_target, s_data_variable_name, s_normalize,
                                 s_num_bins_crossentropy, s_folder_path, device,
                                 normalize=True, load_input_sequence=True, load_target=True, extended_target_size=256, **__
                                 ):
    filtered_data_loader_indecies_dict = filtered_data_loader_indecies[idx]
    file = filtered_data_loader_indecies_dict['file']
    first_idx_input_sequence = filtered_data_loader_indecies_dict['first_idx_input_sequence']
    # The last index is included! (np.arange(1,5) = [1,2,3,4]
    last_idx_input_sequence = filtered_data_loader_indecies_dict['last_idx_input_sequence']
    target_idx_input_sequence = filtered_data_loader_indecies_dict['target_idx_input_sequence']
    input_idx_upper_left = filtered_data_loader_indecies_dict['input_idx_upper_left_h_w']
    data_dataset = xr.open_dataset('{}/{}'.format(s_folder_path, file))

    if load_input_sequence:
        # input_data_set = data_dataset.isel(time=slice(first_idx_input_sequence,
        #                                               last_idx_input_sequence))  # last_idx_input_sequence + 1 like in np! Did I already do that prior?

        input_data_set = data_dataset.isel(time=np.arange(first_idx_input_sequence, last_idx_input_sequence),
                                           x=np.arange(input_idx_upper_left[0], input_idx_upper_left[0]+s_width_height),
                                           y=np.arange(input_idx_upper_left[1], input_idx_upper_left[1]+s_width_height))
        # Using arange leads to same result as slice() (tested)

        input_sequence = input_data_set[s_data_variable_name].values

        # input_sequence = torch.from_numpy(input_sequence)
        # input_sequence = input_sequence.to(device)

        # Get rid of steps dimension (nothing to do with pysteps)
        input_sequence = input_sequence[0, :, :, :]

        if normalize:
            input_sequence = lognormalize_data(input_sequence, mean_filtered_log_data, std_filtered_log_data,
                                               transform_f, s_normalize)
    else:
        input_sequence = None

    if load_target:
        target_data_set = data_dataset.isel(time=target_idx_input_sequence,
                                            x=np.arange(input_idx_upper_left[0], input_idx_upper_left[0]+s_width_height),
                                            y=np.arange(input_idx_upper_left[1], input_idx_upper_left[1]+s_width_height))
        target = target_data_set[s_data_variable_name].values
        del data_dataset
        # Get rid of steps dimension as we only have one index anyways
        target = target[0]

        target = T.CenterCrop(size=extended_target_size)(torch.from_numpy(target))

        if normalize:
            target = lognormalize_data(target, mean_filtered_log_data, std_filtered_log_data, transform_f,
                                       s_normalize)

        target_one_hot = img_one_hot(target, s_num_bins_crossentropy, linspace_binning)
        target_one_hot = einops.rearrange(target_one_hot, 'w h c -> c w h')

        # This ugly bs added for the extended version of target_one_hot required for gaussian smoothing
        target_one_hot_extended = target_one_hot

        target = np.array(T.CenterCrop(size=s_width_height_target)(target))
        target_one_hot = T.CenterCrop(size=s_width_height_target)(target_one_hot)

    else:
        target = None
        target_one_hot = None

    return input_sequence, target_one_hot, target, target_one_hot_extended


def lognormalize_data(data, mean_data, std_data, transform_f, s_normalize):
    """
    We take log first, then do z normalization!
    mean_data and std_data therefore have to be calculated in log space!
    (This has been implemented correctly in filtering_data_scraper)
    """
    data = transform_f(data)
    if s_normalize:
        data, _, _ = normalize_data(data, mean_data=mean_data, std_data=std_data)
    return data


def filtering_data_scraper(transform_f, last_input_rel_idx, target_rel_idx, s_folder_path, s_data_file_name, s_width_height,
                           s_data_variable_name, s_time_span, s_local_machine_mode, s_width_height_target, s_min_rain_ratio_target,
                           s_num_samples_per_frame, s_choose_time_span=False, **__):
    '''
    time span only refers to a single file
    '''

    filtered_data_loader_indecies = []

    # num_x and sum_x as well as sum_x_squared are used to calculate the first and second momentum for data normalization
    num_x = 0  # num_x is the number of data points, meaning the TOTAL NUMBER OF PIXELS, that have passed the filter
    sum_log_x = 0     # sum_x and sum_x squared are the sums of all data points / the squres of all datapoints
    sum_log_x_squared = 0
    sum_x = 0
    sum_x_squared = 0

    num_frames_passed_filter = 0

    num_frames_total = 0

    linspace_binning_min_unnormalized = np.inf
    linspace_binning_max_unnormalized = -np.inf


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: DONT ALLOW s_time_span CHOOSING, SCREWS UP INDECIES WHEN LOADING DATA
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Loading complete data into ram. At the moment 142 GB
    data_sequence_t_h_w = load_data_sequence_preliminary(s_folder_path, s_data_file_name, s_width_height, s_data_variable_name,
                                                   s_choose_time_span, s_time_span, s_local_machine_mode)
    # THIS DATA HAS NANS IN IT!
    # For some reason the 5 min Radolan data has some few negative values that are extremely close to zero. For
    # an analysis see comments of https://3.basecamp.com/5660298/buckets/35200082/messages/7121548207

    data_sequence_t_h_w_not_truncated = data_sequence_t_h_w
    # Cut off all extensive nans at the edges:
    data_sequence_t_h_w, upper_left_truncation_coordinate_h, upper_left_truncation_coordinate_w = truncate_nan_padding(data_sequence_t_h_w)
    upper_left_truncation_coordinate_h_w = np.array((upper_left_truncation_coordinate_h.item(),
                                                         upper_left_truncation_coordinate_w.item()))
    # upper_left_truncation_coordinate give h, w of the upper left truncation point

    # replace all negative data with zeros
    zero_mask = data_sequence_t_h_w < 0
    data_sequence_t_h_w[zero_mask] = 0

    # Iterate through the time steps (up until out of bounds depending on lead time)
    # TODO: !!!!!!! ONLY DEBUG PURPOSE UNDO THIS!!!!!!!!!
    # for i in range(np.shape(data_sequence_t_h_w)[0] - target_rel_idx):
    for i in range(16):

        # Create the height and width for input frames indecies by gridding data_sequence with random offset
        # (new random offset for each input, target chunk)
        input_indecies_upper_left_h_w = gridding_data_sequence_indicies(data_sequence_t_h_w.shape[1:3], s_width_height)

        # iterate through pictures
        for input_idx_upper_left_h_w in input_indecies_upper_left_h_w:

            first_idx_input_sequence = i
            last_idx_input_sequence = i + last_input_rel_idx
            target_idx_input_sequence = i + target_rel_idx

            # Do the cropping based on the upper left pixel
            input_sequence = data_sequence_t_h_w[first_idx_input_sequence:last_idx_input_sequence,
                          input_idx_upper_left_h_w[0]: input_idx_upper_left_h_w[0] + s_width_height,
                          input_idx_upper_left_h_w[1]: input_idx_upper_left_h_w[1] + s_width_height,]

            # For the target we do the cropping based on the wisth and height of the input...
            target = data_sequence_t_h_w[target_idx_input_sequence,
                          input_idx_upper_left_h_w[0]: input_idx_upper_left_h_w[0] + s_width_height,
                          input_idx_upper_left_h_w[1]: input_idx_upper_left_h_w[1] + s_width_height,]

            # Throwing out all frames with nans only in them (in input sequence along time dimension)
            if torch.isnan(target).all() or torch.isnan(input_sequence).all(dim=-2).all(dim=-1).any(dim=0):
                continue

            # ... and then center crop it to the target size
            target = T.CenterCrop(size=s_width_height_target)(target)

            # Set all nans to zero in the input_sequence:
            input_sequence = torch.nan_to_num(input_sequence, nan=0.0)

            # Set all nans to zero in the target.
            # This is only done for filtering purposes! When loading the target we will keep the nans and simply
            # not include them in the loss calculation, such that they do not affect the gradient
            input_sequence = torch.nan_to_num(input_sequence, nan=0.0)


            # TODO: Write all code beloqw for torch instead of numpy
            target = target.cpu().numpy()
            input_sequence = input_sequence.cpu().numpy()

            num_frames_total += 1

            # TODO: !!!!!!! ONLY DEBUG PURPOSE UNDO THIS!!!!!!!!!
            # if filter(input_sequence, target, s_min_rain_ratio_target):
            if True:
                num_frames_passed_filter += 1
                filtered_data_loader_indecies_dict = {}
                filtered_data_loader_indecies_dict['file'] = s_data_file_name
                filtered_data_loader_indecies_dict['first_idx_input_sequence'] = first_idx_input_sequence
                filtered_data_loader_indecies_dict['last_idx_input_sequence'] = last_idx_input_sequence
                filtered_data_loader_indecies_dict['target_idx_input_sequence'] = target_idx_input_sequence
                # input_idx_upper_left_h_w gives the upper left point of our input frame. However this has to be transferred
                # from the truncated coordinate system to the original coordinate system of the data_sequence, therefore
                # we add by upper_left_truncation_coordinate_h_w; plotted this and works as intended!
                filtered_data_loader_indecies_dict['input_idx_upper_left_h_w'] = (
                        np.array(input_idx_upper_left_h_w) + upper_left_truncation_coordinate_h_w)
                filtered_data_loader_indecies_dict['time_span'] = s_time_span if s_choose_time_span else None
                filtered_data_loader_indecies.append(filtered_data_loader_indecies_dict)

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # ! IGNORES FIRST ENTRIES: For means and std to normalize data only the values of the target sequence are taken !
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # We are iterating through all 256x256 target frames that have been accepted by the filter
                # TODO: !!!!! Only normalizing on target frames at the moment !!!!!!
                num_x += np.shape(input_sequence.flatten())[0]
                sum_log_x += np.sum(transform_f(input_sequence.flatten()))
                sum_log_x_squared += np.sum(transform_f(input_sequence.flatten()) ** 2)

                sum_x += np.sum(input_sequence.flatten())
                sum_x_squared += np.sum(input_sequence.flatten() ** 2)

                # linspace binning min and max have to be normalized later as the means and stds are available


                min_input = np.min(input_sequence)
                max_input = np.max(input_sequence)
                # TODO: Could it be that bug with min is introduced because double center cropping as done here
                # yields a different result from single center cropping to target (shift by a pixel or sth?)?
                min_target = np.min(target)
                max_target = np.max(target)

                min_curr_input_and_target = np.min([min_input, min_target])
                max_curr_input_and_target = np.max([max_input, max_target])

                if min_curr_input_and_target < 0:
                    # This should never occur as Filter should filter out all negative values
                    raise Exception('Values smaller than 0 within the test and validation dataset. Probably NaNs')

                # max_curr_input_and_target = np.max(
                #         curr_data_sequence[np.r_[i:last_idx_input_sequence, target_idx_input_sequence]])

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

        # We need to calculate the mean and std of the log data, as we are first taking log, then z normalizing in log space
        mean_filtered_log_data = sum_log_x / num_x
        std_filtered_log_data = np.sqrt((sum_log_x_squared / num_x) - mean_filtered_log_data ** 2)

        # These are the means and stds for the unnormalized data, which we need to select data in certain z ranges
        mean_filtered_data = sum_x / num_x
        std_filtered_data = np.sqrt((sum_x_squared / num_x) - mean_filtered_data ** 2)

    return (filtered_data_loader_indecies, mean_filtered_log_data, std_filtered_log_data, mean_filtered_data, std_filtered_data,
            linspace_binning_min_unnormalized, linspace_binning_max_unnormalized)


def gridding_data_sequence_indicies(data_sequence_height_width: tuple, grid_parcel_height_width: int):
    '''
    This function creates the indecies to get the height and width for the input frames
    (from which also the targets can be created by center cropping)

    A grid is simply layed on top of the data sequence. Each time this function is called, the grid is also
    moived randomly (uniform random) in the magnitude of grid_parcel_height_width

    Output: List of indecies of the upper left corners (height, width) of the grid cells that can be used as the input
    for the network (to get the target simply center crop)

    data_sequence_height_width: (h,w) height and width of the data sequence that the grid is layed on top to
    grid_parcel_height_width: height and width of the grid parcels (one int), corresponds to input height width
    h --> height
    w --> width
    '''

    # Grid offset to move the grip randomly
    grid_offset_h = random.randint(0, grid_parcel_height_width)
    grid_offset_w = random.randint(0, grid_parcel_height_width)

    input_indecies_h_w_upper_left = []

    parcel_num_h = 0
    parcel_num_w = 0

    # Iterating through the height up until out of bounds
    while True:
        index_h_top = grid_offset_h + parcel_num_h * grid_parcel_height_width
        index_h_bottom = index_h_top + grid_parcel_height_width
        parcel_num_h += 1

        if index_h_bottom > data_sequence_height_width[0]:
            break

        # Iterating through the width up until out of bounds
        while True:
            index_w_left = grid_offset_w + parcel_num_w * grid_parcel_height_width
            index_w_right = index_w_left + grid_parcel_height_width
            parcel_num_w += 1

            if index_w_right > data_sequence_height_width[1]:
                break
            else:
                input_indecies_h_w_upper_left.append((index_h_top, index_w_left))

    return input_indecies_h_w_upper_left


def truncate_nan_padding(data_tensor):
    """
    Efficiently truncates the sides of the 3D tensor to remove NaNs without cutting off any non-NaNs.

    Parameters:
    - data_tensor: A torch.Tensor object with shape [time, height, width].

    Returns:
    - A truncated version of data_tensor.
    """
    # Create a mask of non-NaNs
    valid_mask = ~torch.isnan(data_tensor)

    # Aggregate mask along the height and width dimensions to find valid rows and columns
    valid_heights = valid_mask.any(dim=2)  # Check each height for any non-NaN across all widths
    valid_widths = valid_mask.any(dim=1)  # Check each widthumn for any non-NaN across all heights

    # Find the min and max indices for valid heights and widthumns
    min_height_idx = valid_heights.any(dim=0).nonzero().min()
    max_height_idx = valid_heights.any(dim=0).nonzero().max()
    min_width_idx = valid_widths.any(dim=0).nonzero().min()
    max_width_idx = valid_widths.any(dim=0).nonzero().max()

    # Use these indices to slice the tensor, adjust max indices to include the boundary values
    truncated_tensor = data_tensor[:, min_height_idx:max_height_idx+1, min_width_idx:max_width_idx+1]

    return truncated_tensor, min_height_idx, min_width_idx


def calc_class_frequencies(filtered_indecies, linspace_binning, mean_filtered_log_data, std_filtered_log_data, transform_f,
                           settings, s_num_bins_crossentropy, normalize=True, **__):
    '''
    The more often class occurs, the lower the weight value
    TODO: However observed, that classes with lower mean and max precipitation have higher weight ??!!
    '''

    class_count = torch.zeros(s_num_bins_crossentropy, dtype=torch.int64)

    for idx in range(len(filtered_indecies)):
        _, target_one_hot, target, _ = load_input_target_from_index(idx, filtered_indecies, linspace_binning,
                                                                 mean_filtered_log_data, std_filtered_log_data,
                                                                 transform_f,
                                                                 normalize=normalize, load_input_sequence=False,
                                                                 load_target=True, **settings)

        class_count += torch.sum(target_one_hot, (1, 2)).type(torch.int64)

    sample_num = torch.sum(class_count)

    # class_weights = sample_num / class_count
    class_weights = 1 / class_count

    # TODO: How to handle class_count == 0 ? At the moment --> inf
    # However those classes that do not appear are never used, therefore it does not really matter what the
    # Entries for those classes are.
    # Is this correct (from https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2 )

    return class_weights, class_count, sample_num


def class_weights_per_sample(filtered_indecies, class_weights, linspace_binning, mean_filtered_log_data, std_filtered_log_data,
                                transform_f, settings, normalize=True):

    target_mean_weights = []

    for idx in range(len(filtered_indecies)):
        _, target_one_hot, target, _ = load_input_target_from_index(idx, filtered_indecies, linspace_binning,
                                                                 mean_filtered_log_data, std_filtered_log_data,
                                                                 transform_f,
                                                                 normalize=normalize, load_input_sequence=False,
                                                                 load_target=True, **settings)
        # Checked this: Seems to do what it is supposed to (high rain (rare class) corresponds to high weight)
        target_one_hot_indecies = torch.argmax(target_one_hot, dim=0)
        target_class_weighted = class_weights[target_one_hot_indecies]

        mean_weight = torch.mean(target_class_weighted).item()
        target_mean_weights.append(mean_weight)
    return target_mean_weights


def quantile_binning(filtered_indecies, linspace_binning, mean_filtered_log_data, std_filtered_log_data, transform_f,
                           settings, s_num_bins_crossentropy, normalize=True, **__):
    '''
    The more often class occurs, the lower the weight value
    TODO: However observed, that classes with lower mean and max precipitation have higher weight ??!!
    '''

    class_count = torch.zeros(s_num_bins_crossentropy, dtype=torch.int64)

    for idx in range(len(filtered_indecies)):
        _, target_one_hot, target, _ = load_input_target_from_index(idx, filtered_indecies, linspace_binning,
                                                                 mean_filtered_log_data, std_filtered_log_data,
                                                                 transform_f,
                                                                 normalize=normalize, load_input_sequence=False,
                                                                 load_target=True, **settings)

        class_count += torch.sum(target_one_hot, (1, 2)).type(torch.int64)

    sample_num = torch.sum(class_count)

    # class_weights = sample_num / class_count
    class_weights = 1 / class_count

    # TODO: How to handle class_count == 0 ? At the moment --> inf
    # However those classes that do not appear are never used, therefore it does not really matter what the
    # Entries for those classes are.
    # Is this correct (from https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2 )

    return class_weights, class_count, sample_num


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
    '''
    Normalizing data, NO LOG TRANSFORMATION
    '''
    flattened_data = data_sequence.flatten()
    if mean_data is None:
        mean_data = np.mean(flattened_data)
    if std_data is None:
        std_data = np.std(flattened_data)
    return (data_sequence - mean_data) / std_data, mean_data, std_data


def inverse_normalize_data(data_sequence, mean_orig_data, std_orig_data, inverse_log=True, inverse_normalize=True):
    '''
    Assumes that the original data has been logtransformed first and subsequently normalized to standard normal
    Works for torch tensors and numpy arrays
    '''

    if isinstance(data_sequence, torch.Tensor):
        # If input is a torch tensor
        if inverse_normalize:
            data_sequence = data_sequence * std_orig_data + mean_orig_data
        if inverse_log:
            data_sequence = torch.exp(data_sequence) - 1

    elif isinstance(data_sequence, np.ndarray):
        # If input is a numpy array
        if inverse_normalize:
            data_sequence = data_sequence * std_orig_data + mean_orig_data
        if inverse_log:
            data_sequence = np.exp(data_sequence) - 1

    else:
        raise ValueError("Unsupported data type. Please provide a torch tensor or a numpy array.")

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

    # This has to be enabled, when loading the nc netCDF data set
    # data_arr = data_arr[:, 0, :, :]
    data_arr = data_arr[0, :, :, :]

    # Get rid of steps dimension
    # if s_local_machine_mode:
    #     data_arr = data_arr[:, 0, :, :]
    # else:
    #     data_arr = data_arr[0, :, :, :]
    data_tensor = torch.from_numpy(data_arr)

    # Crop --> TODO: Implement this with x y variables of NetCDF in future!
    # Doing cropping in filtering_data_scraper therefore commented out
    # data_tensor = T.CenterCrop(size=s_width_height)(data_tensor)
    return data_tensor


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


def invnorm_linspace_binning(linspace_binning, linspace_binning_max, mean_filtered_log_data, std_filtered_log_data):
    '''
    Inverse normalizes linspace binning
    By default the linspace binning only includes the lower bounds#
    Therefore the highest upper bound is missing which is given by linspace_binning_max
    '''
    linspace_binning_inv_norm = inverse_normalize_data(np.array(linspace_binning), mean_filtered_log_data, std_filtered_log_data)
    linspace_binning_max_inv_norm = inverse_normalize_data(np.array(linspace_binning_max), mean_filtered_log_data, std_filtered_log_data, inverse_log=True)
    return linspace_binning_inv_norm, linspace_binning_max_inv_norm.item()
