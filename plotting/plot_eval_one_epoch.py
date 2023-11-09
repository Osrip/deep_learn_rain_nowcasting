import pandas as pd

from helper.helper_functions import one_hot_to_mm
from load_data import inverse_normalize_data
from baselines import LKBaseline
import torchvision.transforms as T
import numpy as np
import os
from pysteps import verification

import matplotlib.pyplot as plt


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
             ps_multiple_sigmas, fss_logspace_threshold, fss_linspace_scale, prefix='', calc_on_every_n_th_batch = 4, **__):

    '''
    This function calculates the mean and std of the FSS over the whole dataset given by the data loader (validations
    set required). The FSS is calculated for different thresholds and scales. The thresholds are given by the
    fss_logspace_threshold parameter and the scales are given by the fss_linspace_scale parameter. The FSS is calculated
    for each threshold and scale for each sample in the dataset of which the mean and std are then calculated.
    ** expects plot_settings
    Always inv normalizes (independently of ps_inv_normalize) as optical flow cannot operate in inv norm space!
    In progress...

    calc_on_every_n_th_batch=4 <--- Only use every nth batch of data loder to calculate FSS (for speed); at least one
    batch is calculated
    '''
    if calc_on_every_n_th_batch < len(data_loader):
        calc_on_every_n_th_batch = len(data_loader)

    filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized,\
        linspace_binning_max_unnormalized = filter_and_normalization_params

    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params


    inv_norm = lambda x: inverse_normalize_data(x, mean_filtered_data, std_filtered_data, inverse_log=True,
                                                           inverse_normalize=True)

    # thresholds = np.exp(np.linspace(np.log(fss_logspace_threshold[0]), np.log(fss_logspace_threshold[1]), fss_logspace_threshold[2]))
    thresholds = np.linspace(fss_logspace_threshold[0], fss_logspace_threshold[1], fss_logspace_threshold[2])
    # I want this behaviour: np.exp(np.linspace(np.log(0.01), np.log(0.1), 5))
    scales = np.linspace(fss_linspace_scale[0], fss_linspace_scale[1], fss_linspace_scale[2])
    df_data = []



    fss_calc = verification.get_method("FSS")

    for scale in scales:
        for threshold in thresholds:
            fss_model_list_const_param = []
            fss_lk_baseline_list_const_param = []
            for i, (input_sequence, target_one_hot, target, _) in enumerate(data_loader):
                if not (i % calc_on_every_n_th_batch == 0):
                    break

                print('Calculating FSS for sample {} of {}'.format(i, len(data_loader))) #  Debug

                input_sequence = input_sequence.to(ps_device)
                model = model.to(ps_device)
                pred = model(input_sequence)

                if ps_gaussian_smoothing_multiple_sigmas:
                    pred = pred[0].detach().cpu()


                pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)
                del pred
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


# Function to plot the data with the given specifications
def plot_fss(s_dirs, **__):
    # Load the data from the uploaded CSV file
    data = pd.read_csv('{}/fss_None.csv'.format(s_dirs['logs']))

    # Filter unique scales from the data
    scales = data['scale'].unique()

    for scale in scales:
        # Filter data for the current scale
        scale_data = data[data['scale'] == scale]

        # Plot
        fig, ax = plt.subplots()
        ax.plot(scale_data['threshold'], scale_data['fss_lk_baseline_mean'], '--', color='red', label='Baseline Mean')
        ax.fill_between(scale_data['threshold'], scale_data['fss_lk_baseline_mean'] - scale_data['fss_lk_baseline_std'],
                        scale_data['fss_lk_baseline_mean'] + scale_data['fss_lk_baseline_std'], color='red', alpha=0.2)

        ax.plot(scale_data['threshold'], scale_data['fss_model_mean'], '--', color='green', label='Model Mean')
        ax.fill_between(scale_data['threshold'], scale_data['fss_model_mean'] - scale_data['fss_model_std'],
                        scale_data['fss_model_mean'] + scale_data['fss_model_std'], color='green', alpha=0.2)

        ax.set_xlabel('Threshold')
        ax.set_ylabel('FSS Mean')
        # ax.set_xscale('log')
        ax.set_title(f'FSS Mean and Std Dev at Scale {scale}')
        ax.legend()

        # Save the plot with the specified format
        plt.savefig(f'{s_dirs["plot_dir_fss"]}/fss_checkpoint_variable_threshold_scale_{scale}.png')
        plt.show()
        plt.close()  # Close the plot to free memory


def plot_fss_by_threshold(s_dirs, N, **__):
    # Load the data from the uploaded CSV file
    data = pd.read_csv(f"{s_dirs['logs']}/fss_None.csv")

    # Get unique thresholds, sorted
    unique_thresholds = np.sort(data['threshold'].unique())
    # Determine the indices to sample thresholds as evenly as possible
    indices = np.round(np.linspace(0, len(unique_thresholds) - 1, N)).astype(int)
    sampled_thresholds = unique_thresholds[indices]

    for threshold in sampled_thresholds:
        # Filter data for the current threshold
        threshold_data = data[data['threshold'] == threshold]

        # Plot
        fig, ax = plt.subplots()
        ax.plot(threshold_data['scale'], threshold_data['fss_lk_baseline_mean'], '--', color='red', label='Baseline Mean')
        ax.fill_between(threshold_data['scale'],
                        threshold_data['fss_lk_baseline_mean'] - threshold_data['fss_lk_baseline_std'],
                        threshold_data['fss_lk_baseline_mean'] + threshold_data['fss_lk_baseline_std'],
                        color='red', alpha=0.2)

        ax.plot(threshold_data['scale'], threshold_data['fss_model_mean'], '--', color='green', label='Model Mean')
        ax.fill_between(threshold_data['scale'],
                        threshold_data['fss_model_mean'] - threshold_data['fss_model_std'],
                        threshold_data['fss_model_mean'] + threshold_data['fss_model_std'],
                        color='green', alpha=0.2)

        ax.set_xlabel('Scale')
        ax.set_ylabel('FSS Mean')
        ax.set_title(f'FSS Mean and Std Dev at Threshold {threshold:.2f}')
        ax.legend()

        # Save the plot with the specified format
        plt.savefig(f"{s_dirs['plot_dir_fss']}/fss_mean_vs_scale_threshold_{threshold:.2f}.png")
        plt.show()
        plt.close()  # Close the plot to free memory


if __name__ == '__main__':
    run_dir = '/home/jan/jan/programming/first_CNN_on_Radolan/runs/Run_20231108-115128no_gaussian_blurring_with_exp_lr_schedule'

    s_dirs = {}
    s_dirs['plot_dir_fss'] = '{}/plots/fss/'.format(run_dir)
    s_dirs['logs'] = '{}/logs'.format(run_dir)
    plot_fss(s_dirs)
    plot_fss_by_threshold(s_dirs, N=5)








