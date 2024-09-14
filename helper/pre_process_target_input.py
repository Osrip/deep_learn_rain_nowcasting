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


def normalize_data(data, log_mean, log_std):
    '''
    Log(x+1) (log base e), then z - normalization
    log_mean and log_std are 1st and 2nd moments from the log data
    Can handle torch and np data
    '''
    log_mean = float(log_mean)
    log_std = float(log_std)
    if isinstance(data, torch.Tensor):
        return (torch.log1p(data) - log_mean) / log_std
    else:
        return (np.log1p(data) - log_mean) / log_std # log1p takes natural logarithm of x + 1, numerically stable


def inverse_normalize_data(data, log_mean, log_std):
    '''
    Log(x+1) (log base e), then z - normalization
    log_mean and log_std are 1st and 2nd moments from the log data
    Can handle torch and np data
    '''
    if isinstance(data, torch.Tensor):
        return torch.expm1(data * log_std + log_mean)
    else:
        return np.expm1(data * log_std + log_mean)


def invnorm_linspace_binning(linspace_binning, linspace_binning_max, mean_filtered_log_data, std_filtered_log_data):
    '''
    Inverse normalizes linspace binning
    By default the linspace binning only includes the lower bounds#
    Therefore the highest upper bound is missing which is given by linspace_binning_max
    '''
    linspace_binning_inv_norm = inverse_normalize_data(np.array(linspace_binning), mean_filtered_log_data, std_filtered_log_data)
    linspace_binning_max_inv_norm = inverse_normalize_data(np.array(linspace_binning_max), mean_filtered_log_data, std_filtered_log_data)
    return linspace_binning_inv_norm, linspace_binning_max_inv_norm.item()
