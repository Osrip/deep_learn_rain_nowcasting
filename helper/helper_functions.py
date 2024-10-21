import numpy as np
import pickle
import gzip
import os
from os.path import isfile
from shutil import copyfile

import torch
import einops

from helper.pre_process_target_input import img_one_hot, one_hot_to_lognormed_mm


def create_dilation_list(s_width_height, inverse_ratio=4):
    out = []
    en = 1
    while en <= s_width_height / inverse_ratio:
        out.append(en)
        en = en * 2
        if len(out) > 100:
            raise Exception('Caught up')

    return out


def convert_to_binning_and_back(data_arr: torch.Tensor, linspace_binning, linspace_binning_max):
    '''
    This function converts to binning and back to mm
    This way the data exhibits the same discrete properties as the binned data
    '''
    data_one_hot = img_one_hot(data_arr, num_c=len(linspace_binning), linspace_binning=linspace_binning)
    data_one_hot = einops.rearrange(data_one_hot, 'b w h c -> b c w h')
    data_mm = one_hot_to_lognormed_mm(data_one_hot, linspace_binning, channel_dim=1)
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


def save_project_code(save_folder):
    cwd = os.getcwd()

    onlyfiles = []
    for path, subdirs, files in os.walk(cwd):
        for name in files:
            if (isfile(os.path.join(path, name)) and (not 'venv' in path) and (not 'runs' in path) and
                    (name.endswith('.py') or name.endswith('.txt') or name.endswith('.ipynb') or name.endswith('.sh'))):
                onlyfiles.append(os.path.relpath(os.path.join(path, name), cwd))

    for file in onlyfiles:
        save_code(save_folder, file)


def create_save_name_for_data_loader_vars(
        s_folder_path,
        s_local_machine_mode,
        s_save_prefix_data_loader_vars,
        s_num_bins_crossentropy,
        s_linspace_binning_cut_off_unnormalized,
        s_width_height,
        s_crop_data_time_span,
        **__,
):

    if s_local_machine_mode:
        local_machine_str = 'local_machine_dataset'
    else:
        local_machine_str = 'big_dataset'

    if s_crop_data_time_span is None:
        time_crop_str = 'no_time_cropping'
    else:
        time_crop_str = f'cropped_{s_crop_data_time_span[0]}_to_{s_crop_data_time_span[1]}'


    original_file_name = s_folder_path.split('/')[-1]

    return (f'{s_save_prefix_data_loader_vars}_'
            f'{original_file_name}_'
            f'{local_machine_str}_'
            f'{s_num_bins_crossentropy}_'
            f'{s_linspace_binning_cut_off_unnormalized}_'
            f'{s_width_height}_'
            f'{time_crop_str}')


def save_data_loader_vars(data_loader_vars, settings, s_data_loader_vars_path, **__):
    file_name = create_save_name_for_data_loader_vars(**settings)
    save_zipped_pickle(os.path.join(s_data_loader_vars_path, file_name), data_loader_vars)


def load_data_loader_vars(settings, s_data_loader_vars_path, **__):
    file_name = create_save_name_for_data_loader_vars(**settings)
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
