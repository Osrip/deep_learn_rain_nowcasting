from typing import Union

import numpy as np
import torch
import einops
from torch.nn import functional as F


def set_nans_zero(input_sequence: torch.Tensor) -> torch.Tensor:
    '''
    Fast on-the-fly pre-processing of input_sequence in training/validation loop
    Replace NaNs with zeros
    '''
    nan_mask = torch.isnan(input_sequence)
    input_sequence[nan_mask] = 0
    return input_sequence


def pre_process_target_to_one_hot(
        target: torch.Tensor,
        linspace_binning_params,
        s_num_bins_crossentropy,
        **__) -> torch.Tensor:
    '''
    Fast on-the fly pre-processing of target in training / validation loop
    Converting into binned / one-hot target
    Handle nans --> Simply set one hot to zeros at each nan
    '''
    # Creating binned target
    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params
    target_binned = img_one_hot(target, s_num_bins_crossentropy, linspace_binning)
    target_binned = einops.rearrange(target_binned, 'b w h c -> b c w h')
    return target_binned






def img_one_hot(data_arr: torch.Tensor, num_c: int, linspace_binning: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    '''
    Adds one hot encoded channel dimension
    Channel dimension is added as -1st dimension, so rearrange dimensions!
    linspace_binning only includes left bin edges
    for nans simply all bins are set to 0
    The observed values are sorted into the bins as follows:
    left edge <= observed value < right edge
    This is tested in the test case tests/test_img_one_hot
    Handle nans --> Simply set one hot to zeros at each nan
    '''

    data_indexed = bin_to_one_hot_index(data_arr, linspace_binning)  # -0.00000001

    if torch.min(data_indexed) < 0:
        err_message = 'ERROR: ONE HOT ENCODING: data_arr_indexed had values below zero.' \
                      ' min of linspace_binning: {}, (all vals lognormalized) data_arr_index <0: {}'\
            .format(data_arr[data_arr < np.min(linspace_binning)], np.min(linspace_binning), data_indexed[data_indexed<0])
        raise ValueError(err_message)

    else:
        data_hot = F.one_hot(data_indexed.long(), num_c)
        # Handle nans --> Simply set one hot to zeros at each nan
        nan_mask = torch.isnan(data_arr)
        data_hot[nan_mask] = torch.zeros(num_c, dtype=torch.long).to(data_hot.device)

    return data_hot


def bin_to_one_hot_index(mm_data: torch.Tensor, linspace_binning: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(linspace_binning, np.ndarray):
        linspace_binning = torch.from_numpy(linspace_binning).to(mm_data.device)
    # For some reason we need right = True here instead of right = False as in np digitize to get the same behaviour
    indecies = torch.bucketize(mm_data, linspace_binning, right=True) - 1
    return indecies


def one_hot_to_lognormed_mm(one_hot_tensor: torch.Tensor, linspace_binning: Union[torch.Tensor, np.ndarray],
                            channel_dim=1) -> torch.Tensor:
    '''
    THIS IS NOT UNDOING LOGNORMALIZATION (therefore naming)
    Converts one hot data back to precipitation mm data based upon argmax (highest bin wins)
    bin value is lower bin bound (given by bin index in linspace_binning)
    '''
    if isinstance(linspace_binning, np.ndarray):
        linspace_binning = torch.from_numpy(linspace_binning).to(one_hot_tensor.device)
    argmax_indecies = torch.argmax(one_hot_tensor, dim=channel_dim)
    mm_data = linspace_binning[argmax_indecies]
    return mm_data


def lognormalize_data(data, mean_log_data, std_log_data, transform_f, s_normalize):
    """
    We take log first, then do z normalization!
    mean_data and std_data therefore have to be calculated in log space!
    (This has been implemented correctly in filtering_data_scraper)
    """
    data = transform_f(data)
    if s_normalize:
        data = normalize_data(data, mean_data=mean_log_data, std_data=std_log_data)
    return data


def normalize_data(data_sequence, mean_data, std_data):
    '''
    Normalizing data, NO LOG TRANSFORMATION
    '''
    return (data_sequence - mean_data) / std_data


def inverse_normalize_data(data_sequence, mean_log_orig_data, std_log_orig_data, inverse_log=True, inverse_normalize=True):
    '''
    Assumes log - then z normalization:
    Assumes that the original data has been logtransformed first and subsequently normalized to standard normal
    Works for torch tensors and numpy arrays
    ! When inverse_log=True make sure to pass the mean and std of the log transformed data !
    '''

    if isinstance(data_sequence, torch.Tensor):
        # If input is a torch tensor
        if inverse_normalize:
            data_sequence = data_sequence * std_log_orig_data + mean_log_orig_data
        if inverse_log:
            data_sequence = torch.expm1(data_sequence)

    elif isinstance(data_sequence, np.ndarray):
        # If input is a numpy array
        if inverse_normalize:
            data_sequence = data_sequence * std_log_orig_data + mean_log_orig_data
        if inverse_log:
            data_sequence = np.expm1(data_sequence) # more numerically stable than np.exp(data_sequence) - 1

    else:
        raise ValueError("Unsupported data type. Please provide a torch tensor or a numpy array.")

    return data_sequence


def invnorm_linspace_binning(linspace_binning, linspace_binning_max, mean_filtered_log_data, std_filtered_log_data):
    '''
    Inverse normalizes linspace binning
    By default the linspace binning only includes the lower bounds#
    Therefore the highest upper bound is missing which is given by linspace_binning_max
    '''
    linspace_binning_inv_norm = inverse_normalize_data(np.array(linspace_binning), mean_filtered_log_data, std_filtered_log_data)
    linspace_binning_max_inv_norm = inverse_normalize_data(np.array(linspace_binning_max), mean_filtered_log_data, std_filtered_log_data)
    return linspace_binning_inv_norm, linspace_binning_max_inv_norm.item()
