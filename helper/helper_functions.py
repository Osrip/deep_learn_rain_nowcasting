import warnings

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


# def map_mm_to_one_hot_index(mm, num_indecies, mm_min, mm_max):
#     '''
#     !!! OLD !!!
#     This is older version --> Use bin_to_one_hot_index_linear
#     index starts counting at 0 and has max index at max_index --> length of indecies is max_index + 1 !!!
#     '''
#     # TODO: Use logarithmic binning to account for long tailed data distribution of precipitation???
#     # Add tiny number, such that np. floor always accounts for next lower number
#     if mm < mm_min or mm > mm_max:
#         raise IndexError('The input is outside of the given bounds min {} and max {}'.format(mm_min, mm_max))
#     mm_max = mm_max + sys.float_info.min
#     bin_size = (mm_max - mm_min) / num_indecies
#     index = int(np.floor(mm / bin_size))
#     # Covering the case that mm == mm_max in which case np.floor would exceed num indecies
#     if mm_max == mm:
#         index -= 1
#     # np ceil rounds down. In principle we need to round up as all the rest above an integer would fill the next bin
#     # But as the indexing starts at Zero we round down instead
#     return index


def bin_to_one_hot_index(mm_data, linspace_binning):
    '''
    Can directly handle log data
    In practice gets passed log transformed data
    --> linspace_binning_min, linspace_binning_max have to be in log transformed space
    '''
    # Linspace binning always annotates the lowest value of the bin. The very last value (whoich is linspacebinning_max) is
    # not included in the linspace binning, such that the number of entries in linspace binning correstponts to the number of bins
    # Indecies start counting at 1, therefore - 1
    indecies = np.digitize(mm_data, linspace_binning, right=False) - 1
    return indecies


def img_one_hot(data_arr: np.ndarray, num_c: int, linspace_binning) -> torch.Tensor:
    '''
    Adds one hot encoded channel dimension
    Channel dimension is added as -1st dimension, so rearrange dimensions!
    '''
    data_arr_indexed = bin_to_one_hot_index(data_arr, linspace_binning) # -0.00000001


    if np.min(data_arr_indexed) < 0:
        # Handling Values below 0.
        # If warning is encountered fix cause!
        err_message = 'ERROR: ONE HOT ENCODING: data_arr_indexed had values below zero.' \
                      ' Set all vlas < 0 to 0. data_arr < min_linspace_binning: {},' \
                      ' min of linspace_binning: {}, (all vals lognormalized) data_arr_index <0: {}'\
            .format(data_arr[data_arr < np.min(linspace_binning)], np.min(linspace_binning), data_arr_indexed[data_arr_indexed<0])
        warnings.warn(err_message)
        print(err_message)

        print('One hot conversion encountered error --> Set data_arr_indexed[data_arr_indexed < 0] = 0\nTHIS IS BAD, FIX THIS')
        data_arr_indexed[data_arr_indexed < 0] = 0
        data_indexed = torch.from_numpy(data_arr_indexed)
        data_hot = F.one_hot(data_indexed.long(), num_c)

    else:
        data_indexed = torch.from_numpy(data_arr_indexed)
        data_hot = F.one_hot(data_indexed.long(), num_c)

    return data_hot


def one_hot_to_lognorm_mm(one_hot_tensor: torch.Tensor, linspace_binning, linspace_binning_max, channel_dim, mean_bin_vals=False,):
    '''
    THIS IS NOT UNDOING LOGNORMALIZATION
    Converts one hot data back to precipitation mm data based upon argmax (highest bin wins)
    mean_bin_vals==False --> bin value is lower bin bound (given by bin index in linspace_binning)
    mean_bin_vals==True --> bin value is mean of lower and upper bin bound TODO: something better for logspace that binnning is in?
    #TODO --> GEOMETRIC MEAN does not work for lognormal data due to negative values
    #TODO: Calculate arithmetic mena after invlognormalization then lognormalize back
    channel dim: Channel dimension, that represents binning (num channels --> num bins)
    out np.array
    '''

    argmax_indecies = torch.argmax(one_hot_tensor, dim=channel_dim)
    argmax_indecies = argmax_indecies.cpu().detach().numpy()

    if mean_bin_vals:
        linspace_binning_with_max = np.append(linspace_binning, linspace_binning_max)
        mm_lower_bound = linspace_binning[argmax_indecies]
        mm_upper_bound = linspace_binning_with_max[argmax_indecies + 1]
        mm_data = np.mean(np.array([mm_lower_bound, mm_upper_bound]), axis=0)
    else:
        mm_data = linspace_binning[argmax_indecies]
    return mm_data


def convert_to_binning_and_back(data_arr: np.ndarray, linspace_binning, linspace_binning_max):
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


# def save_whole_project(save_folder):
#     '''
#     Copies complete code into simulation folder
#     Todo: does not copy subfolders for some reason!
#     '''
#     cwd = os.getcwd()
#
#     onlyfiles = []
#     for path, subdirs, files in os.walk(cwd):
#         for name in files:
#             if (isfile(join(cwd, name)) and (not 'venv' in path) and (not 'runs' in path) and (name.endswith('.py') or name.endswith('.txt')
#                                                                       or name.endswith('.ipynb') or name.endswith('.sh'))):
#                 onlyfiles.append(os.path.relpath(os.path.join(path, name), cwd))
#
#     # onlyfiles = [f for f in listdir(cwd) if (isfile(join(cwd, f)) and (f.endswith('.py') or f.endswith('.txt') or f.endswith('.ipynb')))]
#     for file in onlyfiles:
#         save_code(save_folder, file)
#
#
# def save_code(save_folder, filename):
#     src = filename
#     dst = save_folder + '/' + src
#     try:
#         copyfile(src, dst)
#     except FileNotFoundError:
#         os.makedirs(dst[0:dst.rfind('/')])


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
                                           s_save_prefix_data_loader_vars, s_num_bins_crossentropy, s_linspace_binning_cut_off_unnormalized, **__):
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

    return '{}_{}_{}_{}_{}_{}_{}'.format(s_save_prefix_data_loader_vars, log_transform_str,
                                      normalize_str, original_file_name, local_machine_str, s_num_bins_crossentropy,
                                         s_linspace_binning_cut_off_unnormalized)


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



