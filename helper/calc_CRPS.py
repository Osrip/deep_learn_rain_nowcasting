import numpy as np
import torch


def crps_vectorized(pred: torch.Tensor, target: torch.Tensor,
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
    bin_edges_right_c_h_w = bin_edges_right[None, :, None, None]
    bin_sizes_unsqueezed_b_c_h_w = bin_sizes[None, :, None, None]

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
    target = target[:, None, :, :]
    heavyside_step = (target <= bin_edges_right_c_h_w).float()

    # Do not use bin weighting as we are operating in lognorm space
    # Without weighting, yields the same results as element-wise,
    # with weighting in both functions we get differing results (no matter wheterh weighting is added before or after cumsum)
    # pred = pred * bin_sizes_unsqueezed_b_c_h_w

    # Calculate CDF
    pred_cdf = torch.cumsum(pred, axis=1)
    # Substract heaviside step
    pred_cdf = pred_cdf - heavyside_step
    # Square
    pred_cdf = torch.square(pred_cdf)

    # Weight according to bin sizes --> When this line is removed, element-wise fails
    # pred_cdf = pred_cdf * bin_sizes_unsqueezed_b_c_h_w

    # Sum to get CRPS --> c dim is summed so b x c x h x w --> b x h x w
    crps = torch.sum(pred_cdf, axis=1)

    return crps

    # Reasoning with Martin
    # TODO: Bullshit to calculate cdf without regarding bin size but then weighting bin size when calculating integral.
    # If we do bin weighting we have to do that before calculating cdf
    # when weighting bins we are calculating CRPS in inverse lognorm space. When not weighting we are calculating in
    # lognorm space --> Xentropy is calculated with equal weights as well. However CRPS is made to be put on original
    # scale


def element_wise_crps(bin_probs, observation, bin_edges, bin_weighting=False):
    """
    VIDEO ZUR IMPLEMENTATION IN ICLOUD NOTES UNTER NOTIZ "CRPS"
    Calculate CRPS between an empirical distribution and a point observation.
    Parameters:
    - bin_edges : array-like, bin edges
    !!including leftmost and rightmost edge!!!
    - bin_probs : array-like, probabilities of each bin --> len(bin_probs) == len(bin_edges - 1) as last right bin not included!
    - observation : float, observed value
    Returns:
    - CRPS value : float
    """
    # TODO Speed this up with jit!
    cdf = np.cumsum(bin_probs)
    crps = 0
    # Iterating through each bin and looking whether observation is outside
    for i in range(len(bin_edges)-1):
        left_edge = bin_edges[i]
        right_edge = bin_edges[i+1]
        if bin_weighting:
            bin_weight = (right_edge - left_edge)
        else:
            bin_weight = 1
        if observation > right_edge:
            crps += cdf[i] ** 2  # * bin_weight  Do not use weighting, we are operating in lognorm space
            # Eveything smaller than observation is added to represent integral

        # elif observation < right_edge:
        else:
            crps += (cdf[i] - 1) ** 2  # * bin_weight
            # For the bin that the observation is in and all larger bins Observation - 1 is added

    return crps


def iterate_crps_element_wise(pred_np, target_inv_normed, linspace_binning_inv_norm, linspace_binning_max_inv_norm):
    '''
    pred_np: binned prediction b x c x h x w
    target_inv_normed: target in inv normed space b x h x w
    linspace_binning_inv_norm: left bins edges in inv normed space
    '''
    calculate_crps_lambda = lambda x, y: element_wise_crps(x, y,
                                                           np.append(linspace_binning_inv_norm, linspace_binning_max_inv_norm),)
    shape_pred = np.shape(pred_np)
    shape_target = np.shape(target_inv_normed)
    if not (shape_target[0] == shape_pred[0] and shape_target[1] == shape_pred[2] and shape_target[2] == shape_pred[3]):
        raise ValueError('Dimensionality mismatch between prediction and target (leaving away channel dimension of prediction')

    crps_out = np.zeros(shape=shape_target)

    for b in range(shape_target[0]):
        for h in range(shape_target[1]):
            for w in range(shape_target[2]):
                crps = calculate_crps_lambda(pred_np[b, :, h, w], target_inv_normed[b, h, w])
                crps_out[b, h, w] = crps
    return crps_out
