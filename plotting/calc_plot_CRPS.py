from helper.helper_functions import one_hot_to_lognorm_mm
from load_data import inverse_normalize_data
import numpy as np
from baselines import LKBaseline
import torch
import torchvision.transforms as T


def calculate_crps(bin_probs, observation, bin_edges):
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
        if observation > right_edge:
            crps += cdf[i] ** 2 * (right_edge - left_edge)
            # Eveything smaller than observation is added to represent integral

        # elif observation < right_edge:
        else:
            crps += (cdf[i] - 1) ** 2 * (right_edge - left_edge)
            # For the bin that the observation is in and all larger bins Observation - 1 is added

    return crps


def calc_CRPS(model, data_loader, filter_and_normalization_params, linspace_binning_params, settings, plot_settings,
             steps_settings,
             ps_runs_path, ps_run_name, ps_checkpoint_name, ps_device, ps_gaussian_smoothing_multiple_sigmas,
             ps_multiple_sigmas, crps_calc_on_every_n_th_batch,
             prefix='', **__):

    '''
    This function calculates the mean and std of the FSS over the whole dataset given by the data loader (validations
    set required). The FSS is calculated for different thresholds and scales. The thresholds are given by the
    fss_space_threshold parameter and the scales are given by the fss_linspace_scale parameter. The FSS is calculated
    for each threshold and scale for each sample in the dataset of which the mean and std are then calculated.
    ** expects plot_settings
    Always inv normalizes (independently of ps_inv_normalize) as optical flow cannot operate in inv norm space!
    In progress...

    calc_on_every_n_th_batch=4 <--- Only use every nth batch of data loder to calculate FSS (for speed); at least one
    batch is calculated
    '''
    if crps_calc_on_every_n_th_batch < len(data_loader):
        crps_calc_on_every_n_th_batch = len(data_loader)

    filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized,\
        linspace_binning_max_unnormalized = filter_and_normalization_params

    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params


    inv_norm = lambda x: inverse_normalize_data(x, mean_filtered_data, std_filtered_data, inverse_log=True,
                                                           inverse_normalize=True)


    df_data = []

    preds_and_targets = {}
    preds_and_targets['pred_mm_inv_normed'] = []
    preds_and_targets['pred_mm_steps_baseline'] = []
    preds_and_targets['target_inv_normed'] = []

    logging_type = None
    steps_baseline = LKBaseline(logging_type, mean_filtered_data, std_filtered_data, use_steps=True,
                                steps_settings=steps_settings, **settings)

    calculate_crps_lambda = lambda x, y: calculate_crps(x, y,
                                                 np.append(linspace_binning_inv_norm, linspace_binning_max_inv_norm), )
    calculate_crps_vec = np.vectorize(calculate_crps_lambda)



    for i, (input_sequence, target_one_hot, target, _) in enumerate(data_loader):
        if not (i % crps_calc_on_every_n_th_batch == 0):
            break

        input_sequence = input_sequence.to(ps_device)
        model = model.to(ps_device)
        pred = model(input_sequence)

        if ps_gaussian_smoothing_multiple_sigmas:
            pred = pred[0].detach().cpu()

        linspace_binning_inv_norm, linspace_binning_max_inv_norm = invnorm_linspace_binning(linspace_binning,
                                                                                            linspace_binning_max,
                                                                                            mean_filtered_data,
                                                                                            std_filtered_data)

        # ! USE INV NORMED PREDICTIONS FROM MODEL ! Baseline is calculated in unnormed space


        # Baseline Prediction
        input_sequence_inv_normed = inv_norm(input_sequence).to('cpu')
        pred_ensemble_steps_baseline, _, _ = steps_baseline(input_sequence_inv_normed)
        pred_ensemble_steps_baseline = T.CenterCrop(size=settings['s_width_height_target'])(pred_ensemble_steps_baseline)


        target = target.detach().cpu().numpy()
        target_inv_normed = inv_norm(target)


        # Calculate CRPS for predictions

        pred_np = pred.cpu().detach().numpy()
        # We take normalized pred as we already passed the inv normalized binning to the calculate_crps function
        crps_np = calculate_crps_vec(pred_np, target_inv_normed)



        preds_and_targets['pred_mm_inv_normed'].append(pred_mm_inv_normed)
        preds_and_targets['pred_mm_steps_baseline'].append(pred_mm_steps_baseline)
        preds_and_targets['target_inv_normed'].append(target_inv_normed)





def invnorm_linspace_binning(linspace_binning, linspace_binning_max, mean_filtered_data, std_filtered_data):
    '''
    Inverse normalizes linspace binning
    By default the linspace binning only includes the lower bounds#
    Therefore the highest upper bound is missing which is given by linspace_binning_max
    '''
    linspace_binning_inv_norm = inverse_normalize_data(np.array(linspace_binning), mean_filtered_data, std_filtered_data)
    linspace_binning_max_inv_norm = inverse_normalize_data(np.array(linspace_binning_max), mean_filtered_data, std_filtered_data, inverse_log=True)
    return linspace_binning_inv_norm, linspace_binning_max_inv_norm.item()


if __name__ == '__main__':
    bin_edges = np.array([0, 1, 2, 3, 4])
    bin_probs = np.array([0, 0, 1, 0])
    observation = 2.5

    test = calculate_crps(bin_probs, observation, bin_edges,)
    pass