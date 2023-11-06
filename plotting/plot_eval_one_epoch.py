import pandas as pd

from helper.helper_functions import one_hot_to_mm
from load_data import inverse_normalize_data
from baselines import LKBaseline
import torchvision.transforms as T
import numpy as np
import os
from pysteps import verification


def plot_CRPS(model, data_loader, filter_and_normalization_params, linspace_binning_params, plot_settings,
              ps_runs_path, ps_run_name, ps_checkpoint_name, ps_device, ps_inv_normalize,
              ps_gaussian_smoothing_multiple_sigmas, ps_multiple_sigmas, prefix='', **__):

    '''
    In progress...
    '''

    filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized,\
        linspace_binning_max_unnormalized = filter_and_normalization_params

    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params

    if ps_inv_normalize:
        inv_norm_or_not = lambda x: inverse_normalize_data(x, mean_filtered_data, std_filtered_data, inverse_log=True,
                                                           inverse_normalize=True)
    else:
        inv_norm_or_not = lambda x: x

    for i, (input_sequence, target_one_hot, target, _) in enumerate(data_loader):
        input_sequence = input_sequence.to(ps_device)
        model = model.to(ps_device)
        pred = model(input_sequence)

        pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)


def calc_FSS(model, data_loader, filter_and_normalization_params, linspace_binning_params, settings, plot_settings,
             ps_runs_path, ps_run_name, ps_checkpoint_name, ps_device, ps_gaussian_smoothing_multiple_sigmas,
             ps_multiple_sigmas, fss_logspace_threshold, fss_linspace_scale, prefix='', **__):

    '''
    ** expects plot_settings
    Always inv normalizes (independently of ps_inv_normalize) as optical flow cannot operate in inv norm space!
    In progress...
    '''

    filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized,\
        linspace_binning_max_unnormalized = filter_and_normalization_params

    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params


    inv_norm = lambda x: inverse_normalize_data(x, mean_filtered_data, std_filtered_data, inverse_log=True,
                                                           inverse_normalize=True)

    thresholds = np.logspace(fss_logspace_threshold[0], fss_logspace_threshold[1], fss_logspace_threshold[2])
    scales = np.linspace(fss_linspace_scale[0], fss_linspace_scale[1], fss_linspace_scale[2])
    df_data = []

    fss_calc = verification.get_method("FSS")

    for scale in scales:
        for threshold in thresholds:
            fss_model_list_const_param = []
            fss_lk_baseline_list_const_param = []
            for i, (input_sequence, target_one_hot, target, _) in enumerate(data_loader):
                input_sequence = input_sequence.to(ps_device)
                model = model.to(ps_device)
                pred = model(input_sequence)

                pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)
                pred_mm_inv_normed = inv_norm(pred_mm)
                # ! USE INV NORMED PREDICTIONS FROM MODEL ! Baseline is calculated in unnormed space

                logging_type = None
                lk_baseline = LKBaseline(logging_type, mean_filtered_data, std_filtered_data, **settings)
                input_sequence_inv_normed = inv_norm(input_sequence).to('cpu')
                pred_mm_lk_baseline, _, _ = lk_baseline(input_sequence_inv_normed)
                pred_mm_lk_baseline = T.CenterCrop(size=32)(pred_mm_lk_baseline)
                pred_mm_lk_baseline = pred_mm_lk_baseline.detach().cpu().numpy()

                target = target.detach().cpu().numpy()
                target_inv_normed = inv_norm(target)

                # Works up until here! predictions of baseline and model have been calculated
                # TODO: Calculate FSS: Write loop that iterates over different thersholds (x-axis) and different scales (several plots or differently colored lines? Or movie?)

                for batch_num in range(np.shape(target_inv_normed)[0]):
                    fss_model = fss_calc(pred_mm_inv_normed[batch_num, :, :], target_inv_normed[batch_num, :, :], threshold, scale)
                    fss_model_list_const_param.append(fss_model)

                    fss_lk_baseline = fss_calc(pred_mm_lk_baseline[batch_num, :, :], target_inv_normed[batch_num, :, :], threshold, scale)
                    fss_lk_baseline_list_const_param.append(fss_lk_baseline)
                    
                # fss = np.nanmean(
                #     [fss(pred_mm_inv_normed[batch_num, :, :], target[batch_num, :, :], threshold, scale)
                #      for batch_num in range(np.shape(target)[0])]
                # )
                # fss_list_const_param.append(fss)

            fss_model_mean = np.nanmean(fss_model_list_const_param)
            fss_model_std = np.nanstd(fss_model_list_const_param)

            fss_lk_baseline_mean = np.nanmean(fss_lk_baseline_list_const_param)
            fss_lk_baseline_std = np.nanstd(fss_lk_baseline_list_const_param)

            df_data.append({'scale': scale, 'threshold': threshold, 'fss_model_mean': fss_model_mean,
                            'fss_model_std': fss_model_std, 'fss_lk_baseline_mean': fss_lk_baseline_mean,
                            'fss_lk_baseline_std': fss_lk_baseline_std})

    df = pd.DataFrame(df_data)

    log_dir = settings['s_dirs']['logs']
    log_name = 'fss{}_{}.csv'.format(prefix, ps_checkpoint_name)
    if not os.path.exists(log_dir):
        # Create a new directory because it does not exist
        os.makedirs(log_dir)

    df.to_csv('{}/{}'.format(log_dir, log_name))
    x=1










