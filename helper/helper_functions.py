import warnings
from typing import Union
import numpy as np
import sys
import pickle
import gzip
import os
from os.path import isfile, join
from shutil import copyfile
import pandas as pd

import torch
from torch.nn import functional as F
import einops


def create_dilation_list(s_width_height, inverse_ratio=4):
    out = []
    en = 1
    while en <= s_width_height / inverse_ratio:
        out.append(en)
        en = en * 2
        if len(out) > 100:
            raise Exception('Caught up')

    return out


def bin_to_one_hot_index(mm_data: torch.Tensor, linspace_binning: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(linspace_binning, np.ndarray):
        linspace_binning = torch.from_numpy(linspace_binning).to(mm_data.device)
    # For some reason we need right = True here instead of right = False as in np digitize to get the same behaviour
    indecies = torch.bucketize(mm_data, linspace_binning, right=True) - 1
    return indecies


def img_one_hot(data_arr: torch.Tensor, num_c: int, linspace_binning: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    '''
    Adds one hot encoded channel dimension
    Channel dimension is added as -1st dimension, so rearrange dimensions!
    linspace_binning only includes left bin edges
    for nans simply all bins are set to 0
    The observed values are sorted into the bins as follows:
    left edge <= observed value < right edge
    This is tested in the test case tests/test_img_one_hot
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


def one_hot_to_lognorm_mm(one_hot_tensor: torch.Tensor, linspace_binning: Union[torch.Tensor, np.ndarray],
                          linspace_binning_max, channel_dim=1) -> torch.Tensor:
    '''
    THIS IS NOT UNDOING LOGNORMALIZATION
    Converts one hot data back to precipitation mm data based upon argmax (highest bin wins)
    bin value is lower bin bound (given by bin index in linspace_binning)
    '''
    if isinstance(linspace_binning, np.ndarray):
        linspace_binning = torch.from_numpy(linspace_binning).to(one_hot_tensor.device)
    argmax_indecies = torch.argmax(one_hot_tensor, dim=channel_dim)
    mm_data = linspace_binning[argmax_indecies]
    return mm_data


def convert_to_binning_and_back(data_arr: torch.Tensor, linspace_binning, linspace_binning_max):
    '''
    This function converts to binning and back to mm
    This way the data exhibits the same discrete properties as the binned data
    '''
    data_one_hot = img_one_hot(data_arr, num_c=len(linspace_binning), linspace_binning=linspace_binning)
    data_one_hot = einops.rearrange(data_one_hot, 'b w h c -> b c w h')
    data_mm = one_hot_to_lognorm_mm(data_one_hot, linspace_binning, linspace_binning_max, channel_dim=1)
    return data_mm


def save_zipped_pickle(title, data):
    '''
    Compresses data and saves it
    '''
    with gzip.GzipFile(title + '.pickle.pgz', 'w') as f:
        pickle.dump(data, f)


def load_zipped_pickle(file):
    data = gzip.GzipFile(file + '.pickle.pgz', 'rb')
    data = pickle.load(data)
    return data


def save_dict_pickle_csv(title, save_dict):
    with open('{}.csv'. format(title), 'w') as f:
        for key in save_dict.keys():
            f.write("%s,%s\n" % (key, save_dict[key]))
    save_zipped_pickle('{}'.format(title), save_dict)


def save_tuple_pickle_csv(save_dict, folder, file_name):
    with open('{}/{}.csv'. format(folder, file_name), 'w') as f:
        for en in save_dict:
            f.write("%s\n" % (en))
    save_zipped_pickle('{}/{}'.format(folder, file_name), save_dict)


def save_whole_project(save_folder):
    cwd = os.getcwd()

    onlyfiles = []
    for path, subdirs, files in os.walk(cwd):
        for name in files:
            if (isfile(os.path.join(path, name)) and (not 'venv' in path) and (not 'runs' in path) and
                    (name.endswith('.py') or name.endswith('.txt') or name.endswith('.ipynb') or name.endswith('.sh'))):
                onlyfiles.append(os.path.relpath(os.path.join(path, name), cwd))

    for file in onlyfiles:
        save_code(save_folder, file)


def _create_save_name_for_data_loader_vars(s_folder_path, s_log_transform, s_normalize, s_local_machine_mode,
                                           s_save_prefix_data_loader_vars, s_num_bins_crossentropy, s_linspace_binning_cut_off_unnormalized,
                                           s_width_height, **__):
    if s_log_transform:
        log_transform_str = 'log_transform'
    else:
        log_transform_str = 'no_log_transform'

    if s_normalize:
        normalize_str = 'normalize'
    else:
        normalize_str = 'no_normalize'

    if s_local_machine_mode:
        local_machine_str = 'local_machine_dataset'
    else:
        local_machine_str = 'big_dataset'


    original_file_name = s_folder_path.split('/')[-1]

    return '{}_{}_{}_{}_{}_{}_{}_{}'.format(s_save_prefix_data_loader_vars, log_transform_str,
                                      normalize_str, original_file_name, local_machine_str, s_num_bins_crossentropy,
                                         s_linspace_binning_cut_off_unnormalized, s_width_height)


def save_data_loader_vars(data_loader_vars, settings, s_data_loader_vars_path, **__):
    file_name = _create_save_name_for_data_loader_vars(**settings)
    save_zipped_pickle(os.path.join(s_data_loader_vars_path, file_name), data_loader_vars)


def load_data_loader_vars(settings, s_data_loader_vars_path, **__):
    file_name = _create_save_name_for_data_loader_vars(**settings)
    path = os.path.join(s_data_loader_vars_path, file_name)
    if not os.path.exists('{}.pickle.pgz'.format(path)):
        print('No data loader vars found at {}, therefore filtering data from scratch'.format(path))
        raise FileNotFoundError('File {} not found'.format(path))
    print('Loading data loader vars from {}'.format(path))
    return load_zipped_pickle(path)


def save_code(save_folder, filename):
    src = filename
    dst = os.path.join(save_folder, src)  # Properly join the folder and filename
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)  # Create intermediate directories if they don't exist
        copyfile(src, dst)
    except FileNotFoundError:
        os.makedirs(dst[0:dst.rfind('/')])


def convert_tensor_to_np(tensor):
    return tensor.cpu().detach().numpy()


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def flatten_list(lst):
    '''
    Flattens list by one dimension
    '''
    return [item for sublist in lst for item in sublist]


def no_special_characters(str):
    str_new = ''
    for char in str:
        if char.isalnum() or (char == '_'):
            str_new += char
        else:
            str_new += '_'
    return str_new


def df_cols_to_list_of_lists(keys, df):
    out_list = []
    for key in keys:
        out_list.append(df[key].to_list())
    return out_list


def convert_list_of_lists_to_lists_of_lists_with_means(list_of_lists):
    mean_f = lambda x: np.mean(x)
    return [[mean_f(l)] for l in list_of_lists]
