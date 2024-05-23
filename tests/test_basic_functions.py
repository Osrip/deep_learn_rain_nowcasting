from helper.helper_functions import one_hot_to_lognorm_mm, img_one_hot, bin_to_one_hot_index
from load_data import normalize_data, inverse_normalize_data
import numpy as np
import torch
import einops


def test_img_one_hot():
    '''
    Unit test img_one_hot()
    Expected behavior is that values are sorted in between the bin borders
    here given by 1, 2, 3, 4, 5 (4 bins)
    For example value 1.3 would be sorted into
    bin 1 as
    1 (left bound) <= 1.3 (observed value) < 2 (right bound)
    '''
    test_data = torch.Tensor([1, 1.5, 2, 2.1, 3, 3.9, 4, 5])
    one_hot_control = torch.Tensor(
        [[1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 1]])
    num_c = 4
    linspace_binning = np.linspace(torch.min(test_data).item(), torch.max(test_data).item(),
                                   num=num_c,
                                   endpoint=False)
    data_hot = img_one_hot(test_data, num_c, linspace_binning)
    # expected linspace binning: array([1. , 1.8, 2.6, 3.4, 4.2])
    assert (data_hot == one_hot_control).all()


def test_one_hot_converting():
    '''
    Integration test
    Test converting to one hot img_one_hot and then nack to mm with one_hot_to_mm in both, mean_bins and lower bound options
    (Validation data set is binning directly from test_data_set --> Information lost due to bins, that can't be reproduced)
    '''
    # Generate a tensor with standard normal distribution
    test_data = torch.randn(2, 256, 256)

    num_c = 64
    linspace_binning_min = torch.min(test_data).item()
    linspace_binning_max = torch.max(test_data).item()

    linspace_binning = np.linspace(linspace_binning_min, linspace_binning_max,
                                   num=num_c,
                                   endpoint=False)

    # Test conversion with img_one_hot...
    data_hot = img_one_hot(test_data, num_c, linspace_binning)
    # Put channel dimension where it belongs
    data_hot = einops.rearrange(data_hot, 'i w h c -> i c w h')

    is_one_hot(data_hot, one_hot_dim=1)

    # ... then back with one_hot_to_mm assigning lower bin bound
    data_binned_lower_bound_mm = one_hot_to_lognorm_mm(data_hot, linspace_binning, linspace_binning_max=linspace_binning_max,
                                                       channel_dim=1)

    # Sort the test_data directly into bins as valiadation. Take the lower bound of the bin as value for the bin
    # Hacking some stuff with lambda and np.vecotrize to enable an additional argument (linspace_binning)
    # in the vectorized function as np.vectorize does not allow this by default
    next_smallest_func = lambda x: next_smallest(x, linspace_binning)
    v_next_smallest = np.vectorize(next_smallest_func)
    validate_data_lower_bound = v_next_smallest(test_data)
    tolerance = 1e-3
    assert torch.isclose(data_binned_lower_bound_mm, torch.tensor(validate_data_lower_bound), atol=tolerance).all()
    #
    #
    # # ... this time converting from one hot to mm with one_hot_to_mm assigning mean bin value
    # data_binned_mean_of_bounds_mm = one_hot_to_lognorm_mm(data_hot, linspace_binning, linspace_binning_max=linspace_binning_max,
    #                                                       channel_dim=1, mean_bin_vals=True)
    # mean_of_bounds_func = lambda x: mean_of_bounds(x, linspace_binning, linspace_binning_max)
    # v_mean_of_bounds = np.vectorize(mean_of_bounds_func)
    # validate_data_mean_bounds = v_mean_of_bounds(test_data)
    # assert (data_binned_mean_of_bounds_mm == validate_data_mean_bounds).all()

def test_torch_bin_to_one_hot_index():

    # Define the two functions to be tested
    def bin_to_one_hot_index_np(mm_data, linspace_binning):
        indecies = np.digitize(mm_data, linspace_binning, right=False) - 1
        return indecies

    bin_to_one_hot_index_torch = bin_to_one_hot_index

    # Test data
    mm_data_np = np.array([0.0, 0.1, 0.5, 0.9, 1.5, 2.0, 2.5])
    linspace_binning_np = np.array([0, 1, 2])
    mm_data_torch = torch.tensor(mm_data_np)
    linspace_binning_torch = torch.tensor(linspace_binning_np)
    # Expected device
    device = 'cpu'
    # Execute both functions
    result_np = bin_to_one_hot_index_np(mm_data_np, linspace_binning_np)
    result_torch = bin_to_one_hot_index_torch(mm_data_torch, linspace_binning_torch)
    # Convert Torch result to NumPy for comparison
    result_torch_np = result_torch.numpy()
    # Compare
    assert np.allclose(result_np, result_torch_np)


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
    Integration test checking whether inverse_normalize can reconstruct the data that has been normalized by normalize()
    '''
    test_data_set = torch.randn(5, 256, 256) * 5 + 432
    mean = torch.mean(test_data_set).item()
    std = torch.std(test_data_set).item()
    normalized_test_data = normalize_data(test_data_set, mean, std)
    reconstructed_test_data = inverse_normalize_data(normalized_test_data, mean, std, inverse_log=False)
    assert (reconstructed_test_data == test_data_set).all()


def test_normalize_inverse_normalize_log():
    '''
    Integration test checking whether inverse_normalize can reconstruct the data that has been normalized by normlaiz()
    '''
    test_data_set = np.random.rand(5, 256, 256) * 5 + 432
    log_test_data = np.log(test_data_set+1)
    mean = np.mean(log_test_data)
    std = np.std(log_test_data)
    normalized_test_data = normalize_data(log_test_data, mean, std)
    reconstructed_test_data = inverse_normalize_data(normalized_test_data, mean, std, inverse_log=True)
    assert (np.round(reconstructed_test_data, 5) == np.round(test_data_set, 5)).all()


def is_one_hot(tensor, one_hot_dim=0):
    assert torch.all(torch.sum(tensor, dim=one_hot_dim) == 1)

def test_all():
    '''
    A mix of unit and integration tests that cover crucial functions
    '''
    test_img_one_hot()
    test_one_hot_converting()
    test_normalize_inverse_normalize()
    test_normalize_inverse_normalize_log()
    test_torch_bin_to_one_hot_index()
    print('All tests successfull')


if __name__ == '__main__':

    test_all()


