import numpy as np
import sys
import pickle
import gzip
import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile

import torch


def create_dilation_list(s_width_height, inverse_ratio=4):
    out = []
    en = 1
    while en <= s_width_height / inverse_ratio:
        out.append(en)
        en = en * 2
        if len(out) > 100:
            # TODO: preliminary, delete this after testing!
            raise Exception('Caught up')

    return out


def smoothing_one_hot_to_dist(one_hot_frame):
    pass


def convolution_no_channel_sum(input, kernel):
    '''
    This is an implementation of a convolution operation using a single filter (iterated only once per b dim), that skips
    summing the results of all channels. This way the output is not a channel dim of size 1 but instead the channel dim is preserved
    '''
    pad = kernel
    torch.nn.functional.pad(input)

def map_mm_to_one_hot_index(mm, num_indecies, mm_min, mm_max):
    '''
    !!! OLD !!!
    This is older version --> Use bin_to_one_hot_index_linear
    index starts counting at 0 and has max index at max_index --> length of indecies is max_index + 1 !!!
    '''
    # TODO: Use logarithmic binning to account for long tailed data distribution of precipitation???
    # Add tiny number, such that np. floor always accounts for next lower number
    if mm < mm_min or mm > mm_max:
        raise IndexError('The input is outside of the given bounds min {} and max {}'.format(mm_min, mm_max))
    mm_max = mm_max + sys.float_info.min
    bin_size = (mm_max - mm_min) / num_indecies
    index = int(np.floor(mm / bin_size))
    # Covering the case that mm == mm_max in which case np.floor would exceed num indecies
    if mm_max == mm:
        index -= 1
    # np ceil rounds down. In principle we need to round up as all the rest above an integer would fill the next bin
    # But as the indexing starts at Zero we round down instead
    return index


def bin_to_one_hot_index_linear(mm_data, linspace_binning):
    '''
    Can directly handle log data
    In practice gets passed log transformed data
    --> linspace_binning_min, linspace_binning_max have to be in log transformed space
    '''
    # TODO: Use logarithmic binning to account for long tailed data distribution of precipitation???
    # Linspace binning always annotates the lowest value of the bin. The very last value (whoich is linspacebinning_max) is not
    # not included in the linspace binning, such that the number of entries in linspace binning correstponts to the number of bins
    # Indecies start counting at 1, therefore - 1
    indecies = np.digitize(mm_data, linspace_binning, right=False) - 1
    return indecies


def one_hot_to_mm(one_hot_tensor, linspace_binning, linspace_binning_max, channel_dim, mean_bin_vals=True):
    '''
    THIS IS NOT UNDOING LOGNORMALIZATION
    Converts one hot data back to precipitation mm data based upon argmax (highest bin wins)
    mean_bin_vals==False --> bin value is lower bin bound (given by bin index in linspace_binning)
    mean_bin_vals==True --> bin value is mean of lower and upper bin bound
    channel dim: Channel dimension, that represents binning (num channels --> num bins)
    '''

    argmax_indecies = torch.argmax(one_hot_tensor, dim=channel_dim)
    argmax_indecies = np.array(argmax_indecies.cpu())
    if mean_bin_vals:
        linspace_binning_with_max = np.append(linspace_binning, linspace_binning_max)
        mm_lower_bound = linspace_binning[argmax_indecies]
        mm_upper_bound = linspace_binning_with_max[argmax_indecies + 1]
        mm_data = np.mean(np.array([mm_lower_bound, mm_upper_bound]), axis=0)
    else:
        mm_data = linspace_binning[argmax_indecies]
    return mm_data





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


def save_dict_pickle_csv(save_dict, folder, file_name):
    with open('{}/{}.csv'. format(folder, file_name), 'w') as f:
        for key in save_dict.keys():
            f.write("%s,%s\n" % (key, save_dict[key]))
    save_zipped_pickle('{}/{}'.format(folder, file_name), save_dict)


def save_tuple_pickle_csv(save_dict, folder, file_name):
    with open('{}/{}.csv'. format(folder, file_name), 'w') as f:
        for en in save_dict:
            f.write("%s\n" % (en))
    save_zipped_pickle('{}/{}'.format(folder, file_name), save_dict)


def save_whole_project(save_folder):
    '''
    Copies complete code into simulation folder
    Todo: does not copy subfolders for some reason!
    '''
    cwd = os.getcwd()

    onlyfiles = []
    for path, subdirs, files in os.walk(cwd):
        for name in files:
            if (isfile(join(cwd, name)) and (not 'venv' in path) and (not 'runs' in path) and (name.endswith('.py') or name.endswith('.txt')
                                                                      or name.endswith('.ipynb') or name.endswith('.sh'))):
                onlyfiles.append(os.path.relpath(os.path.join(path, name), cwd))

    # onlyfiles = [f for f in listdir(cwd) if (isfile(join(cwd, f)) and (f.endswith('.py') or f.endswith('.txt') or f.endswith('.ipynb')))]
    for file in onlyfiles:
        save_code(save_folder, file)


def save_code(save_folder, filename):
    src = filename
    dst = save_folder + '/' + src
    try:
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
