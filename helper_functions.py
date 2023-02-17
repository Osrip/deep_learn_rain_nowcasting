import numpy as np
import sys
from exceptions import BoundsException


def create_dilation_list(width_height, inverse_ratio=4):
    out = []
    en = 1
    while en <= width_height / inverse_ratio:
        out.append(en)
        en = en * 2
        if len(out) > 100:
            # TODO: preliminary, delete this after testing!
            raise Exception('Caught up')

    return out


def map_mm_to_one_hot_index(mm, num_indecies, mm_min, mm_max):
    '''
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

