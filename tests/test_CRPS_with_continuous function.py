import numpy as np
from scipy.special import erf
from scipy.stats import norm
import matplotlib.pyplot as plt
from helper.calc_CRPS import crps_vectorized, element_wise_crps
import torch


def crps_gaussian(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """CRPS for Gaussian distribution.

    Args:
        x (np.ndarray): Ground truth data of size (n_samples, n_features)
        mu (np.ndarray): Mean of size (n_samples, n_features)
        std (np.ndarray): Standard deviation of size (n_samples, n_features)

    Returns:
        crps (np.ndarray): CRPS of size (n_samples, n_features)
    """
    sqrtPi = np.sqrt(np.pi)
    z = (x - mu) / std
    phi = np.exp(-z ** 2 / 2) / (np.sqrt(2) * sqrtPi) #standard normal pdf
    crps = std * (z * erf(z / np.sqrt(2)) + 2 * phi - 1 / sqrtPi) #crps as per Gneiting et al 2005
    return crps


def create_gaussian_binning(mu: np.ndarray, std: np.ndarray) -> (np.ndarray, np.ndarray):
    gauss_random_vars = norm.rvs(size=100000000, loc=mu, scale=std)
    bin_edges_left = np.linspace(np.min(gauss_random_vars), np.max(gauss_random_vars), num=10000, endpoint=False)
    bin_edge_max_right = np.max(gauss_random_vars)
    bin_edges = np.append(bin_edges_left, bin_edge_max_right)
    indecies = np.digitize(gauss_random_vars, bin_edges_left, right=False) - 1
    binning = np.bincount(indecies)
    binning_softmax = binning / np.sum(binning)

    plt.figure()
    plt.plot(binning_softmax)
    plt.title('randomly sampled gauss pdf')
    plt.xlabel('Bin #')
    plt.show()

    return bin_edges, binning_softmax


def vec_crps_gaussian(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        bin_edges, binning_softmax = create_gaussian_binning(mu, std)
        x = torch.tensor(x)
        binning_softmax = torch.from_numpy(binning_softmax)

        pred = binning_softmax
        target = x
        crps = crps_vectorized_one_dim_input(pred=pred , target=target,
                               linspace_binning_inv_norm=bin_edges[0:-1], linspace_binning_max_inv_norm=bin_edges[-1],
                               device='cpu')
        return crps.cpu().numpy()


def element_crps_gaussian(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    bin_edges, binning_softmax = create_gaussian_binning(mu, std)
    return element_wise_crps(bin_probs=binning_softmax, observation=x, bin_edges=bin_edges)


def crps_vectorized_one_dim_input(pred: torch.Tensor, target: torch.Tensor,
                    linspace_binning_inv_norm: np.ndarray, linspace_binning_max_inv_norm: np.ndarray, device, **__):
    '''
    TODO: WHAT HAPPENS IF TARGET IS A DITRIBUTION INSTEAD OF ONE HOT VALUE? --> Not possible as target has no c dimension
    TODO: THIS IS IN PRINCIPLE POSSIBLE TO CALCULATE CRPS IN THAT SCENARIO BUT DOES OUR FUNCTION DEAL WITH THIS CORRECTLY?
    --> TODO: We can calculate BRIER Score independently for each bin. In one hot target case the step function is the same
    TODO: But in case of target distribution it would be different according to the value of the current bin in the target (???)
    pred: pred_np: binned prediction b x c x h x w
    target: target_inv_normed: target in inv normed space b x h x w
    linspace_binning_inv_norm: left bins edges in inv normed space

    returns CRPS for each pixel in shape b x h x w
    '''
    # Calculations related to binning
    bin_edges_all = np.append(linspace_binning_inv_norm, linspace_binning_max_inv_norm)
    bin_edges_all = torch.from_numpy(bin_edges_all).to(device)
    bin_edges_right = bin_edges_all[1:]
    bin_sizes = torch.diff(bin_edges_all)

    # Unsqueeze binning (adding None dimension) to get same dimensionality as pred with c being dim 1
    bin_edges_right_c_h_w = bin_edges_right
    bin_sizes_unsqueezed_b_c_h_w = bin_sizes

    # in element-wise we are looping through b ins while observation stays constant
    # We are adding a new c dimension to targets where for each target value is replaced by an array of 1s and 0s depending on whether
    # the binning is smaller or bigger than the target
    # heavyside step b x c x h x w --> same as pred
    # target b x h x w
    # binning c

    # Adding c dim that are comparisons to observation: Can also be interpreted as
    # Calculate the heavyside step function (0 below a vertain value 1 above)
    # This repaces if condition in element-wise calculation by just adding
    # heavyside_step is -1 for all bin edges that are on the right side (bigger) than the observation (target)
    target = target.unsqueeze(0)
    heavyside_step = (target <= bin_edges_right_c_h_w).float()

    # Calculate CDF
    pred_cdf = torch.cumsum(pred, axis=0)
    # Substract heaviside step
    pred_cdf = pred_cdf - heavyside_step
    # Square
    pred_cdf = torch.square(pred_cdf)
    # Weight according to bin sizes
    pred_cdf = pred_cdf * bin_sizes_unsqueezed_b_c_h_w
    # Sum to get CRPS --> c dim is summed so b x c x h x w --> b x h x w
    crps = torch.sum(pred_cdf, axis=0)

    return crps


if __name__ == '__main__':
    x = 20 #3
    mu = 27# 4
    std = 500 # 2

    vec_crps = vec_crps_gaussian(x, mu, std)
    elem_crps = element_crps_gaussian(x, mu, std)

    test_crps = crps_gaussian(x, mu, std)
    print(f'analytical crps: {test_crps}\nvectorized crps: {vec_crps}\nelement wise crps: {elem_crps}')
    pass
# def pdf_gaussian(x, mu, std):
#     'Create gaussian PDF'
#     sqrtPi = np.sqrt(np.pi)
#     z = (x - mu) / std
#     phi = np.exp(-z ** 2 / 2) / (np.sqrt(2) * sqrtPi) #standard normal pdf
#     return phi
#
#
# def crps_continuous(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
#     sqrtPi = np.sqrt(np.pi)
#
#     phi = pdf_gaussian(x, mu, std)
#     z = (x - mu) / std
#     crps = std * (z * erf(z / np.sqrt(2)) + 2 * phi - 1 / sqrtPi) #crps as per Gneiting et al 2005


