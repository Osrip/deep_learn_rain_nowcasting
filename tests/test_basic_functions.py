from helper_functions import one_hot_to_mm
from load_data import img_one_hot, normalize_data, inverse_normalize_data
import numpy as np
import torch
import einops



def test_img_one_hot():
    '''
    Unittest img_one_hot()
    '''
    test_data = [1, 2, 3, 4, 5]
    one_hot_control = torch.tensor(
        [[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]])
    data_hot, linspace_binning = img_one_hot(test_data, 5, np.min(test_data), np.max(test_data))
    # expected linspace binning: array([1. , 1.8, 2.6, 3.4, 4.2])
    assert (data_hot == one_hot_control).all()


def test_one_hot_converting():
    '''
    Integration test
    Test converting to one hot img_one_hot and then nack to mm with one_hot_to_mm in both, mean_bins and lower bound options
    (Validation data set is binning directly from test_data_set --> Information lost due to bins, that can't be reproduced)
    '''
    test_data = np.random.standard_normal(size=(2, 256, 256))
    num_c = 64
    linspace_binning_min = np.min(test_data)
    linspace_binning_max = np.max(test_data)

    # Test conversion with img_one_hot...
    data_hot, linspace_binning = img_one_hot(test_data, num_c, linspace_binning_min, linspace_binning_max)
    # Put channel dimension where it belongs
    data_hot = einops.rearrange(data_hot, 'i w h c -> i c w h')

    # ... then back with one_hot_to_mm assigning lower bin bound
    data_binned_lower_bound_mm = one_hot_to_mm(data_hot, linspace_binning, linspace_binning_max=linspace_binning_max,
                                               channel_dim=1, mean_bin_vals=False)

    # Sort the test_data directly into bins as valiadation. Take the lower bound of the bin as value for the bin
    # Hacking some stuff with lambda and np.vecotrize to enable an additional argument (linspace_binning)
    # in the vectorized function as np.vectorize does not allow this by default
    next_smallest_func = lambda x: next_smallest(x, linspace_binning)
    v_next_smallest = np.vectorize(next_smallest_func)
    validate_data_lower_bound = v_next_smallest(test_data)
    assert (data_binned_lower_bound_mm == validate_data_lower_bound).all()


    # ... this time converting from one hot to mm with one_hot_to_mm assigning mean bin value
    data_binned_mean_of_bounds_mm = one_hot_to_mm(data_hot, linspace_binning, linspace_binning_max=linspace_binning_max,
                                               channel_dim=1, mean_bin_vals=True)
    mean_of_bounds_func = lambda x: mean_of_bounds(x, linspace_binning, linspace_binning_max)
    v_mean_of_bounds = np.vectorize(mean_of_bounds_func)
    validate_data_mean_bounds = v_mean_of_bounds(test_data)
    assert (data_binned_mean_of_bounds_mm == validate_data_mean_bounds).all()


def next_smallest(x, linspace_binning):
    '''
    Helper for est_one_hot_converting
    '''
    dists = x - linspace_binning
    dists[dists < 0] = np.inf
    return linspace_binning[np.argmin(dists)]


def mean_of_bounds(x, linspace_binning, linspace_binning_max):
    '''
    Helper for est_one_hot_converting
    '''
    linspace_binning_with_max = np.append(linspace_binning, linspace_binning_max)
    dists = x - linspace_binning
    dists[dists < 0] = np.inf
    lower_bounds = linspace_binning[np.argmin(dists)]
    upper_bounds = linspace_binning_with_max[np.argmin(dists)+1]
    return np.mean(np.array([lower_bounds, upper_bounds]), axis=0)


def test_normalize_inverse_normalize():
    '''
    Integration test checking whether inverse_normalize can reconstruct the data that has been normalized by normlaiz()
    '''
    test_data_set = np.random.rand(5,256,256)*5+432
    normalized_test_data, mean, std = normalize_data(test_data_set)
    reconstructed_test_data = inverse_normalize_data(normalized_test_data, mean, std)
    assert (reconstructed_test_data == test_data_set).all()


def test_all():
    '''
    A mix of unit and integration tests that cover crucial functions
    '''
    test_img_one_hot()
    test_one_hot_converting()
    test_normalize_inverse_normalize()
    print('All tests successfull')

if __name__ == '__main__':

    test_all()


