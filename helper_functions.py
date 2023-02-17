import numpy as np
from warnings import WarningMessage


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


def map_mm_to_one_hot_index(mm, max_index, mm_min, mm_max):
    '''
    index starts counting at 0 and has max index at max_index --> length of indecies is max_index + 1 !!!
    '''
    # TODO: Use logarithmic binning to account for long tailed data distribution of precipitation???
    bin_size = (mm_max - mm_min) / (max_index + 1)
    index = int(np.ceil(mm / bin_size))
    # np ceil rounds down. In principle we need to round up as all the rest above an integer would fill the next bin
    # But as the indexing starts at Zero we round down instead
    return index

