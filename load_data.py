# import h5py
import xarray as xr
import numpy as np
import torch
import torchvision.transforms as T
from helper.helper_functions import chunk_list, flatten_list
from helper.pre_process_target_input import img_one_hot, normalize_data
import datetime
from exceptions import CountException
from torch.utils.data import Dataset
import einops
import random
import warnings
from typing import Callable


# Remember to install package netCDF4 !!

class PrecipitationFilteredDataset(Dataset):
    def __init__(self, filtered_data_loader_indecies, mean_filtered_log_data, std_filtered_log_data, linspace_binning_min,
                 linspace_binning_max, linspace_binning, transform_f, settings,
                 s_num_bins_crossentropy, s_folder_path, s_width_height, s_width_height_target, s_data_variable_name,
                 s_local_machine_mode, s_gaussian_smoothing_target, s_gaussian_smoothing_multiple_sigmas, s_data_file_name,
                 device, s_normalize=True, **__):
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

        self.settings = settings
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
        self.s_gaussian_smoothing_target = s_gaussian_smoothing_target
        self.s_gaussian_smoothing_multiple_sigmas = s_gaussian_smoothing_multiple_sigmas
        self.s_device = device

        # Initialize xr dataset
        self.xr_dataset = xr.open_dataset('{}/{}'.format(s_folder_path, s_data_file_name), chunks=None)

    def __len__(self):
        return len(self.filtered_data_loader_indecies)

    def __getitem__(self, idx) -> tuple[torch.Tensor]:
        # Loading everything directly from the disc

        input_sequence = load_input_from_index(idx,
                                               self.xr_dataset,
                                               self.filtered_data_loader_indecies,
                                               self.mean_filtered_log_data,
                                               self.std_filtered_log_data,
                                               self.transform_f,
                                               normalize=True,
                                               **self.settings,)

        target = load_target_from_index(idx,
                                        self.xr_dataset,
                                        self.filtered_data_loader_indecies,
                                        self.mean_filtered_log_data,
                                        self.std_filtered_log_data,
                                        self.transform_f,
                                        normalize=True,
                                        **self.settings,)

        # This is automatically converted into a torch.Tensor by lightning
        return input_sequence, target


def load_input_from_index(idx: int,
                          xr_dataset: xr.Dataset,
                          filtered_data_loader_indecies: list[dict],
                          mean_filtered_log_data: float,
                          std_filtered_log_data: float,
                          transform_f: Callable,
                          s_normalize, s_data_variable_name, s_width_height,
                          normalize=True, **__) -> np.array:
    '''
    This loads the input_sequence directly from the zarr file efficiently from the filtered indecies list that includes spatial
    and temporal indecies.
    idx: the number of the sample
    '''

    # Get the indecies
    filtered_data_loader_indecies_dict = filtered_data_loader_indecies[idx]
    first_idx_input_sequence = filtered_data_loader_indecies_dict['first_idx_input_sequence']
    # The last index is included! (np.arange(1,5) = [1,2,3,4]
    last_idx_input_sequence = filtered_data_loader_indecies_dict['last_idx_input_sequence']
    input_idx_upper_left = filtered_data_loader_indecies_dict['input_idx_upper_left_h_w']

    # !!!! ATTENTION Y --> HEIGHT --> Dim [0]  x--> WIDTH --> Dim [1]
    # TESTED, NOW cuts the expected crop!
    input_data_set = xr_dataset.isel(time=np.arange(first_idx_input_sequence, last_idx_input_sequence+1),
                                       y=np.arange(input_idx_upper_left[0], input_idx_upper_left[0]+s_width_height),
                                       x=np.arange(input_idx_upper_left[1], input_idx_upper_left[1]+s_width_height))
    # Using arange leads to same result as slice() (tested)

    input_sequence = input_data_set[s_data_variable_name].values

    # Get rid of steps dimension (nothing to do with pysteps)
    input_sequence = input_sequence[0, :, :, :]

    # setting all nans to zero (this is only done to input sequence, not to target!)
    nan_mask = np.isnan(input_sequence)
    input_sequence[nan_mask] = 0

    if normalize:
        input_sequence = normalize_data(input_sequence, mean_filtered_log_data, std_filtered_log_data)
    return input_sequence


def load_target_from_index(idx: int,
                           xr_dataset: xr.Dataset,
                           filtered_data_loader_indecies: list[dict],
                           mean_filtered_log_data: float,
                           std_filtered_log_data: float,
                           transform_f: Callable,
                           s_normalize, s_data_variable_name, s_width_height, s_width_height_target,
                           s_gaussian_smoothing_target,
                           normalize=True, extended_target_size=256, **__) -> torch.Tensor:

    # Get the indecies
    filtered_data_loader_indecies_dict = filtered_data_loader_indecies[idx]
    target_idx_input_sequence = filtered_data_loader_indecies_dict['target_idx_input_sequence']
    # Input idx upper left is needed as a spatial reference for the target
    input_idx_upper_left = filtered_data_loader_indecies_dict['input_idx_upper_left_h_w']

    target_data_set = xr_dataset.isel(time=target_idx_input_sequence,
                                      y=np.arange(input_idx_upper_left[0], input_idx_upper_left[0] + s_width_height),
                                      x=np.arange(input_idx_upper_left[1], input_idx_upper_left[1] + s_width_height))
    target = target_data_set[s_data_variable_name].values
    # Get rid of steps dimension as we only have one index anyway
    target = target[0]

    with torch.no_grad():
        # TODO: ONLY LOAD TARGET OF SHAPE s_width_height here or s_width_height_extended
        # TODO Make sure that extended only gets the required paddding for the gaussian kernel!!
        target = torch.from_numpy(target)
        # target = target.to('cpu')

        # extended target is only needed in case of DLBD.
        # Cut either right to default target size or extended target size
        if s_gaussian_smoothing_target:
            target = T.CenterCrop(size=extended_target_size)(target)
        else:
            target = T.CenterCrop(size=s_width_height_target)(target)

        if normalize:
            target = normalize_data(target, mean_filtered_log_data, std_filtered_log_data)
        target = target.detach().cpu().numpy()
    return target


# TODO: !!!! rewrite this such that it only loads extended target if we are really doing gausian smoothing! !!!
def load_input_target_from_index(idx, xr_dataset, filtered_data_loader_indecies, mean_filtered_log_data,
                                 std_filtered_log_data, transform_f, s_width_height, s_width_height_target, s_data_variable_name,
                                 s_normalize, s_gaussian_smoothing_target, s_gaussian_smoothing_multiple_sigmas, device,
                                 normalize=True, load_input_sequence=True, load_target=True, extended_target_size=256, **__
                                 ):
    """
    This loads input and target (controllable via flags)
    Mainly used by data set's __getitem__
    However purposely not a method of the data set as it is also used during data pre-processing for
    calculating oversampling weights
    """

    filtered_data_loader_indecies_dict = filtered_data_loader_indecies[idx]

    first_idx_input_sequence = filtered_data_loader_indecies_dict['first_idx_input_sequence']
    # The last index is included! (np.arange(1,5) = [1,2,3,4]
    last_idx_input_sequence = filtered_data_loader_indecies_dict['last_idx_input_sequence']
    target_idx_input_sequence = filtered_data_loader_indecies_dict['target_idx_input_sequence']
    input_idx_upper_left = filtered_data_loader_indecies_dict['input_idx_upper_left_h_w']

    if load_input_sequence:

        # !!!! ATTENTION Y --> HEIGHT --> Dim [0]  x--> WIDTH --> Dim [1]
        # TESTED, NOW cuts the expected crop!
        input_data_set = xr_dataset.isel(time=np.arange(first_idx_input_sequence, last_idx_input_sequence+1),
                                           y=np.arange(input_idx_upper_left[0], input_idx_upper_left[0]+s_width_height),
                                           x=np.arange(input_idx_upper_left[1], input_idx_upper_left[1]+s_width_height))
        # Using arange leads to same result as slice() (tested)

        input_sequence = input_data_set[s_data_variable_name].values

        # Get rid of steps dimension (nothing to do with pysteps)
        input_sequence = input_sequence[0, :, :, :]

        # setting all nan s to zero (this is only done to input sequence, not to target!)
        nan_mask = np.isnan(input_sequence)
        input_sequence[nan_mask] = 0

        if normalize:
            input_sequence = normalize_data(input_sequence, mean_filtered_log_data, std_filtered_log_data)
    else:
        input_sequence = None

    if load_target:
        # TODO: ONLY LOAD TARGET OF SHAPE s_width_height here or s_width_height_extended
        # TODO Make sure that extended only gets the required paddding for the gaussian kernel!!
        target_data_set = xr_dataset.isel(time=target_idx_input_sequence,
                                            y=np.arange(input_idx_upper_left[0], input_idx_upper_left[0]+s_width_height),
                                            x=np.arange(input_idx_upper_left[1], input_idx_upper_left[1]+s_width_height))
        target = target_data_set[s_data_variable_name].values
        # Get rid of steps dimension as we only have one index anyway
        target = target[0]

        with torch.no_grad():
            target = torch.from_numpy(target)
            target = target.to(device)

            # extended target is only needed in case of DLBD.
            # Cut either right to default target size or extended target size
            if s_gaussian_smoothing_target or s_gaussian_smoothing_multiple_sigmas:
                target = T.CenterCrop(size=extended_target_size)(target)
            else:
                target = T.CenterCrop(size=s_width_height_target)(target)

            if normalize:
                target = normalize_data(target, mean_filtered_log_data, std_filtered_log_data)
    else:
        target = torch.Tensor([])

    # In case of s_gaussian_smoothing_target==True this returns the extended target size
    return input_sequence, target


def create_and_filter_patches(
        s_width_height,
        s_width_height_target,
        s_num_input_time_steps,
        s_num_lead_time_steps,
        s_folder_path,
        s_data_file_name,
        **__):


    y_target, x_target = s_width_height_target, s_width_height_target  # 73, 137 # how many pixels in y and x direction
    y_input, x_input = s_width_height, s_width_height
    y_input_padding, x_input_padding = 32, 32 # Additional padding that the frames that will be returned to data loader get for Augmentation

    # Filter conditions:
    threshold_mm_rain_each_pixel = 0.1  # threshold for each pixel filter condition
    threshold_percentage_pixels = 0.5

    # Loading data into xarray
    load_path = '{}/{}'.format(s_folder_path, s_data_file_name)
    data = xr.open_dataset(load_path, engine='zarr')
    data = data.squeeze()
    # Cut off the beginning  of the data as the size of the data chunk, that one sample has (input frames + lead time + target)
    data_shortened = data.isel(
        time=slice(s_num_input_time_steps + s_num_lead_time_steps, -1)
    )

    # partition the data into pt x y_target x x_target blocks using coarsen --> construct DatasetCoarsen object
    # In this implementation each target corresponds to one patch
    coarse = data_shortened.coarsen(
        y=y_target,
        x=x_target,
        # time = 1, # TODO: This way we are making patches with 4 subsequent time frames. This way we are only taking a target every 'pt'th time step
        side="left",  # "left" means that the blocks are aligned to the left of the input
        boundary="trim")  # boundary="trim" removes the last block if it is too small

    # construct a new data set, where the patches are folded into a new dimension
    patches = coarse.construct(
        # time = ("time_outer", "time_inner"),
        y=("y_outer", "y_inner"),
        x=("x_outer", "x_inner"))
    # Replace NaNs with 0s for the filter (we do not have to do this! Makes it less likely for the edge cases to occur in the target.)
    # We also have the option to filter for NaNs in the input to completely prohibit edge cases
    patches_no_nan = patches.fillna(0)

    # --- FILTER ---
    # define a threshold for each pixel --> we get a pixel-wise boo
    patches_boolean_pixelwise = patches_no_nan > threshold_mm_rain_each_pixel
    # We are calculating the percentage of pixels that passed filter (mean of boolean gives percentage of True)
    # --> we are getting rid of the patch dimensions y_inner and x_inner,
    patches_percentage_pixels_passed = patches_boolean_pixelwise.mean(("y_inner", "x_inner"))  # , "time_inner"
    # Now we are creating a boolean again by comparing the percentage of pixels that exceed the rain threshold to the minimally required
    # percentage of pixels that exceed the rain threshold
    # --> valid_patches includes only the y_outer and x_outer, where each pair of indecies represents one patch. Its values are boolean, indicating whether or not the outer indecies correspond to a valid or invalid patch.
    valid_patches_boo = patches_percentage_pixels_passed > threshold_percentage_pixels

    # get the outer coordinates for all valid blocks (valid_time, valid_x, valid_y)
    # (valid_patches_boo is boolean, np.nonzero returns the indecies of the pixels that are non-zero, thus True)
    valid_target_indecies_outer = np.array(np.nonzero(valid_patches_boo.RV_recalc.values)).T

    return valid_target_indecies_outer


def filtering_data_scraper(transform_f, s_folder_path, s_data_file_name, s_width_height,
                           s_data_variable_name, s_width_height_target,
                           s_data_preprocessing_chunk_num, s_num_input_time_steps, s_num_lead_time_steps,
                           s_max_num_filter_hits, **__):
    '''
    This huge ass function is doing all the filtering of the data and returns a list of indecies for the final data that
    is used for training and validation (both temporal and spatial indecies). This way the data can be directly loaded
    from the original data set using these indecies by the dataloader
    What does this function do?
    - Processes the data in chunks, that get laoded into the memory (Number of chunks given by s_data_preprocessing_chunk_num,
      which has to be adjusted to available memory
    - Iterates through all 5 min time steps, then processes all input frames and the target that belong to this time step
    - For each of these input - target segments a gridding with random offset is applied to create crops of the input size (s_width_height)
      (only one random grid per segment / time step in the data, such that there is no overlap between the crops in one time step
    - Iterates through these crops (all crops with nans only are removed)

    In the upcoming steps all nans (both input and target are treated as zeros)
    (! Attention ! For final training input nans are set to zero and target nans are disregarded for loss calc,
     as one-hot is set to [0,0,0,...] (as of March 2024))
    - Filter condition is applied (currently to the target only) - nans treated as zero
    - For all cropped input-target segments that pass the filter condition, the spatio-temporal indecies are returned
      (upper left corner of crop with respect to the input width / height given by s_width_height),
      additionally the normalization parameters are calculated based on the targets of the filtered cropped segments

    For output data format look at the comments at return statement

    This torch - only implementation has been tested against the old torch - numpy mix implementation and yields
    the same results (commit b5ce5eef5dd3a085f040fa562832898b05e19218 and
    test function: https://chatgpt.com/share/c0a7f3d3-e091-45bd-8ca3-ff316071b132)
    '''

    filtered_data_loader_indecies = []

    # num_x and sum_x as well as sum_x_squared are used to calculate the first and second momentum for data normalization
    num_x = 0  # num_x is the number of data points, meaning the TOTAL NUMBER OF PIXELS, that have passed the filter
    sum_log_x = torch.tensor(0.0)     # sum_x and sum_x squared are the sums of all data points / the squres of all datapoints
    sum_log_x_squared = torch.tensor(0.0)
    sum_x = torch.tensor(0.0)
    sum_x_squared = torch.tensor(0.0)

    num_frames_passed_filter = 0

    num_frames_total = 0

    linspace_binning_min_unnormalized = torch.inf
    linspace_binning_max_unnormalized = -torch.inf

    # Loading data into xarray
    load_path = '{}/{}'.format(s_folder_path, s_data_file_name)
    print('Loading training/validation data from {}'.format(load_path))
    dataset = xr.open_dataset(load_path, chunks=None)  # , chunks=None according to Sebastian more efficient as it avoids dask (default is chunks=1)
    num_time_steps = dataset.sizes['time']

    # chunk_time_indecies gives start and stop indecies in time dimension
    chunk_time_indecies, chunk_size = create_chunk_indices(num_time_steps, s_data_preprocessing_chunk_num)

    # Iterate through CHUNKS, that get loaded into ram
    for chunk_num in range(len(chunk_time_indecies) - 1):
        chunk_start_idx = chunk_time_indecies[chunk_num]
        chunk_end_idx = chunk_time_indecies[chunk_num + 1]
        dataset_chunk = dataset.isel(time=slice(chunk_start_idx, chunk_end_idx))
        # Load into ram
        data_chunk_t_h_w_np = dataset_chunk[s_data_variable_name].values

        with (torch.no_grad()):
            # Convert into torch tensor ON CPU
            data_chunk_t_h_w = torch.from_numpy(data_chunk_t_h_w_np).to('cpu')

            # Get rid of prediction steps dimension
            data_chunk_t_h_w = data_chunk_t_h_w[0, :, :, :]

            # THIS DATA HAS NANS IN IT!
            # For some reason the 5 min non-dwd Radolan data has some few negative values that are extremely close to zero. For
            # an analysis see comments of https://3.basecamp.com/5660298/buckets/35200082/messages/7121548207

            data_sequence_t_h_w_not_truncated = data_chunk_t_h_w
            # Cut off all extensive nans at the edges:
            data_chunk_t_h_w, upper_left_truncation_coordinate_h, upper_left_truncation_coordinate_w = truncate_nan_padding(data_chunk_t_h_w)
            upper_left_truncation_coordinate_h_w = np.array((upper_left_truncation_coordinate_h.item(),
                                                                 upper_left_truncation_coordinate_w.item()))
            # upper_left_truncation_coordinate give h, w of the upper left truncation point

            # replace all negative data with zeros
            zero_mask = data_chunk_t_h_w < 0
            data_chunk_t_h_w[zero_mask] = 0
            if zero_mask.any():
                warnings.warn(f'There are values below zero in data_set,'
                              f' lowest value: {torch.min(data_chunk_t_h_w[zero_mask])}')

            # Iterate through the time steps (up until out of bounds depending on lead time)
            total_lead = s_num_lead_time_steps + s_num_input_time_steps
            if (np.shape(data_chunk_t_h_w)[0] - total_lead) < 1:
                raise ValueError(f'Preprocessing failed! Reduce s_data_preprocessing_chunk_num!'
                                 f' Chunk size is {chunk_size}, whereas '
                                 f'total lead (s_num_lead_time_steps + s_num_input_time_steps) is {total_lead}.')

            # iterate through TIME - For each time step in the data set an input - target segment is created
            for time_idx_in_chunk in range(np.shape(data_chunk_t_h_w)[0] - total_lead):

                # Create the height and width for input frames indecies by gridding data_sequence with random offset
                # (new random offset for each input, target chunk)
                input_indecies_upper_left_h_w = gridding_data_sequence_indecies(data_chunk_t_h_w.shape[1:3],
                                                                                s_width_height)

                # iterate through CROPS of the grid
                for input_idx_upper_left_h_w in input_indecies_upper_left_h_w:

                    # Calculate the indecies relative to the start of the chunk:
                    first_idx_chunk = time_idx_in_chunk
                    last_idx_chunk = first_idx_chunk + s_num_input_time_steps - 1
                    target_idx_chunk = last_idx_chunk + s_num_lead_time_steps

                    # Calculate the indecies relative to the whole data set
                    first_idx_input_sequence = time_idx_in_chunk + chunk_size * chunk_num
                    last_idx_input_sequence = first_idx_input_sequence + s_num_input_time_steps - 1
                    target_idx_input_sequence = last_idx_input_sequence + s_num_lead_time_steps

                    # Do the cropping based on the upper left pixel
                    input_sequence = data_chunk_t_h_w[first_idx_chunk:last_idx_chunk+1,
                                  input_idx_upper_left_h_w[0]: input_idx_upper_left_h_w[0] + s_width_height,
                                  input_idx_upper_left_h_w[1]: input_idx_upper_left_h_w[1] + s_width_height,]

                    # For the target we do the cropping based on the wisth and height of the input...
                    target = data_chunk_t_h_w[target_idx_chunk,
                                  input_idx_upper_left_h_w[0]: input_idx_upper_left_h_w[0] + s_width_height,
                                  input_idx_upper_left_h_w[1]: input_idx_upper_left_h_w[1] + s_width_height,]

                    # Throwing out all frames with nans only in them (in input sequence along time dimension)
                    if torch.isnan(target).all() or torch.isnan(input_sequence).all(dim=-2).all(dim=-1).any(dim=0):
                        continue  # Skip code below and proceed right to next loop iteration

                    # if torch.isnan(target).sum().item() / target.numel() > 0.05:
                    #     continue

                    # ... and then center crop it to the target size
                    target = T.CenterCrop(size=s_width_height_target)(target)

                    # Set all nans to zero in the input_sequence:
                    input_sequence = torch.nan_to_num(input_sequence, nan=0.0)

                    # Set all nans to zero in the target.
                    # This is only done for filtering purposes! When loading the target we will keep the nans and simply
                    # not include them in the loss calculation, such that they do not affect the gradient

                    target = torch.nan_to_num(target, nan=0.0)

                    # The code below has been transformed from np to torch!
                    num_frames_total += 1

                    # Filter data
                    if filter(input_sequence, target):
                        num_frames_passed_filter += 1

                        filtered_data_loader_indecies_dict = {}
                        filtered_data_loader_indecies_dict['first_idx_input_sequence'] = first_idx_input_sequence
                        filtered_data_loader_indecies_dict['last_idx_input_sequence'] = last_idx_input_sequence
                        filtered_data_loader_indecies_dict['target_idx_input_sequence'] = target_idx_input_sequence
                        # input_idx_upper_left_h_w gives the upper left point of our input frame. However this has to be transferred
                        # from the truncated coordinate system to the original coordinate system of the data_sequence, therefore
                        # we add by upper_left_truncation_coordinate_h_w; plotted this and works as intended!
                        filtered_data_loader_indecies_dict['input_idx_upper_left_h_w'] = (
                                np.array(input_idx_upper_left_h_w) + upper_left_truncation_coordinate_h_w)
                        filtered_data_loader_indecies_dict['input_idx_upper_left_h_w_cropped_coordinates_DEBUG'] = np.array(input_idx_upper_left_h_w)
                        filtered_data_loader_indecies.append(filtered_data_loader_indecies_dict)

                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # ! IGNORES FIRST ENTRIES: For means and std to normalize data only the values of the target sequence are taken !
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        # We are iterating through all 256x256 target frames that have been accepted by the filter
                        # TODO: !!!!! Only normalizing on target frames at the moment !!!!!!
                        num_x += input_sequence.flatten().shape[0]
                        sum_log_x += torch.sum(transform_f(input_sequence.flatten()))
                        sum_log_x_squared += torch.sum(transform_f(input_sequence.flatten()) ** 2)

                        sum_x += torch.sum(input_sequence.flatten())
                        sum_x_squared += torch.sum(input_sequence.flatten() ** 2)

                        # linspace binning min and max have to be normalized later as the means and stds are available

                        min_input = torch.min(input_sequence)
                        max_input = torch.max(input_sequence)
                        min_target = torch.min(target)
                        max_target = torch.max(target)

                        min_curr_input_and_target = min(min_input, min_target)
                        max_curr_input_and_target = max(max_input, max_target)

                        if min_curr_input_and_target < 0:
                            # This should never occur as Filter should filter out all negative values
                            raise Exception('Values smaller than 0 within the test and validation dataset. Probably NaNs')

                        if linspace_binning_min_unnormalized > min_curr_input_and_target:
                            linspace_binning_min_unnormalized = min_curr_input_and_target

                        if linspace_binning_max_unnormalized < max_curr_input_and_target:
                            linspace_binning_max_unnormalized = max_curr_input_and_target

                        # This limits number of hits according to num_frames_passed_filter (introduced for local mode)
                        if s_max_num_filter_hits:
                            if num_frames_passed_filter >= s_max_num_filter_hits:
                                break
                    if s_max_num_filter_hits:
                        if num_frames_passed_filter >= s_max_num_filter_hits:
                            break
                if s_max_num_filter_hits:
                    if num_frames_passed_filter >= s_max_num_filter_hits:
                        break
            if s_max_num_filter_hits:
                if num_frames_passed_filter >= s_max_num_filter_hits:
                    break

    # TODO: Is Bessel's correction (+1 accounting for extra degree of freedom) needed here (not a big effect)?
    if num_x == 0:
        raise Exception('No data passed the filter conditions such that there is no '
                        'data for training and validation.')
    else:
        print(f'{num_frames_passed_filter} data points out of a total of {num_frames_total} scanned data points'
              ' passed the filter condition')

    # We need to calculate the mean and std of the log data, as we are first taking log, then z normalizing in log space
    mean_filtered_log_data = sum_log_x / num_x
    std_filtered_log_data = torch.sqrt((sum_log_x_squared / num_x) - mean_filtered_log_data ** 2)

    # These are the means and stds for the unnormalized data, which we need to select data in certain z ranges
    mean_filtered_data = sum_x / num_x
    std_filtered_data = torch.sqrt((sum_x_squared / num_x) - mean_filtered_data ** 2)

    mean_filtered_log_data = mean_filtered_log_data.item()
    std_filtered_log_data = std_filtered_log_data.item()
    mean_filtered_data = mean_filtered_data.item()
    std_filtered_data = std_filtered_data.item()
    linspace_binning_min_unnormalized = linspace_binning_min_unnormalized.item()
    linspace_binning_max_unnormalized = linspace_binning_max_unnormalized.item()

    return (filtered_data_loader_indecies,  # list of dict, each list entry corresponds to one sample
            # dict keys: first_idx_input_sequence: int, last_idx_input_sequence: int, target_idx_input_sequence: int,
            # input_idx_upper_left_h_w: np.array of length 2
            mean_filtered_log_data, std_filtered_log_data, # float, float
            mean_filtered_data, std_filtered_data, # float, float
            linspace_binning_min_unnormalized, linspace_binning_max_unnormalized) # float, float


def create_chunk_indices(num_time_steps, num_chunks):
    """
    Splits num_time_steps into num_chunks equally sized chunks (as much as possible).
    Returns an array where each pair of consecutive elements represent the start (inclusive)
    and end (exclusive) indices for slicing with xarray's isel.

    Parameters:
    - num_time_steps: Total number of time steps to split.
    - num_chunks: Number of chunks to split the time steps into.

    Returns:
    - A list of indices for direct use with isel slice().
    """
    # Calculate chunk size
    chunk_size = num_time_steps // num_chunks
    remainder = num_time_steps % num_chunks

    indices = [0]  # Start with the initial index
    for i in range(num_chunks):
        # Calculate the next start index
        next_index = indices[-1] + chunk_size + (1 if i < remainder else 0)
        indices.append(next_index)

    return indices, chunk_size


def gridding_data_sequence_indecies(data_sequence_height_width: tuple, grid_parcel_height_width: int):
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

    # Find the min and max indices for valid heights and widths
    min_height_idx = valid_heights.any(dim=0).nonzero().min()
    max_height_idx = valid_heights.any(dim=0).nonzero().max()
    min_width_idx = valid_widths.any(dim=0).nonzero().min()
    max_width_idx = valid_widths.any(dim=0).nonzero().max()

    # Use these indices to slice the tensor, adjust max indices to include the boundary values
    truncated_tensor = data_tensor[:, min_height_idx:max_height_idx+1, min_width_idx:max_width_idx+1]

    return truncated_tensor, min_height_idx, min_width_idx


def calc_class_frequencies(filtered_indecies, linspace_binning, mean_filtered_log_data, std_filtered_log_data, transform_f,
                           settings, s_num_bins_crossentropy, s_folder_path, s_data_file_name, device, normalize=True,
                           **__):
    '''
    The more often a class occurs, the lower the weight value : class_weights = 1 / class_count
    TODO: However observed, that classes with lower mean and max precipitation have higher weight ??!!
    '''
    xr_dataset = xr.open_dataset('{}/{}'.format(s_folder_path, s_data_file_name))
    class_count = torch.zeros(s_num_bins_crossentropy, dtype=torch.int64).to(device)

    for idx in range(len(filtered_indecies)):
        target = load_target_from_index(
            idx,
            xr_dataset,
            filtered_indecies,
            mean_filtered_log_data,
            std_filtered_log_data,
            transform_f,
            normalize=True,
            **settings)

        target = torch.from_numpy(target).to(device)
        target_one_hot = img_one_hot(target, s_num_bins_crossentropy, linspace_binning)
        target_one_hot = einops.rearrange(target_one_hot, 'w h c -> c w h')

        class_count += torch.sum(target_one_hot, (1, 2)).type(torch.int64)

    sample_num = torch.sum(class_count)

    # class_weights = sample_num / class_count
    class_weights = 1 / class_count

    # How to handle class_count == 0 ? At the moment --> inf
    # However those classes that do not appear are never used, therefore it does not really matter what the
    # Entries for those classes are.
    # Is this correct (from https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2 )

    return class_weights, class_count, sample_num


def class_weights_per_sample(filtered_indecies, class_weights, linspace_binning, mean_filtered_log_data, std_filtered_log_data,
                                transform_f, settings, s_folder_path, s_data_file_name, s_num_bins_crossentropy, device, **__):

    target_mean_weights = []
    xr_dataset = xr.open_dataset('{}/{}'.format(s_folder_path, s_data_file_name))

    for idx in range(len(filtered_indecies)):
        target = load_target_from_index(idx,
                                         xr_dataset,
                                         filtered_indecies,
                                         mean_filtered_log_data,
                                         std_filtered_log_data,
                                         transform_f,
                                         normalize=True,
                                         **settings)

        target = torch.from_numpy(target).to(device)
        target_one_hot = img_one_hot(target, s_num_bins_crossentropy, linspace_binning)
        target_one_hot = einops.rearrange(target_one_hot, 'w h c -> c w h')

        # Checked this: Seems to do what it is supposed to (high rain (rare class) corresponds to high weight)
        target_one_hot_indecies = torch.argmax(target_one_hot, dim=0)
        target_class_weighted = class_weights[target_one_hot_indecies]

        mean_weight = torch.mean(target_class_weighted).item()
        target_mean_weights.append(mean_weight)
    return target_mean_weights


def filter(
        input_sequence: torch.Tensor,
        target: torch.Tensor,
        percentage=0.5,
        min_amount_rain=0.2):

    '''
    Previously: min_amount_rain=0.05
    In 50% of the target there has to be at least some rain (>0.2 mm),
    No values below zero in target or input_sequence
    reasonable amount of data passes: percentage=0.5, min_amount_rain=0.05
    '''
    if (target[target > min_amount_rain].numel() > percentage * target.numel()):
        return True
    else:
        return False


def random_splitting_filtered_indecies(indecies, num_training_samples, chunk_size):
    '''
    This randomly splits training and validation data
    Chunk size: Size that consecutive data is chunked in when performing random splitting
    '''
    chunked_indecies = chunk_list(indecies, chunk_size)
    num_chunks_training = int(num_training_samples / chunk_size)

    chunk_idxs = np.arange(len(chunked_indecies))
    np.random.shuffle(chunk_idxs)
    training_chunks = [chunked_indecies[i] for i in chunk_idxs[0:num_chunks_training]]
    vaildation_chunks = [chunked_indecies[i] for i in chunk_idxs[num_chunks_training:]]
    training_indecies = flatten_list(training_chunks)
    validation_indecies = flatten_list(vaildation_chunks)
    return training_indecies, validation_indecies







