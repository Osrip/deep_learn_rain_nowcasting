
import numpy as np
import torch
import os

from helper.checkpoint_handling import load_from_checkpoint, create_data_loaders, load_data_from_run
# from helper.helper_functions import load_zipped_pickle
from plotting.plot_snapshots import plot_snapshots
from plotting.calc_plot_CRPS import calc_CRPS, plot_crps
from plotting.calc_plot_FSS import calc_FSS, plot_fss_by_scales, plot_fss_by_threshold,\
    plot_fss_by_threshold_one_plot, plot_fss_by_scales_one_plot
from plotting.plot_spread_skill_ratio import plot_spread_skill


def plot_from_checkpoint(checkpoint_name,
                         plot_fss_settings,
                         plot_crps_settings,
                         steps_settings,
                         plot_settings,
                         ps_runs_path,
                         ps_run_name,
                         ps_plot_snapshots,
                         ps_plot_fss,
                         ps_plot_crps,
                         ps_plot_spread_skill,
                         **__):
    '''
    Loads model from corresponding epoch and plotsthings up
    This does a forward pass! GPU resources required!
    '''

    settings, filtered_indecies_training, filtered_indecies_validation, linspace_binning_params, filter_and_normalization_params \
        = load_data_from_run(ps_runs_path, ps_run_name)

    # Cover case of old versions, where crps loss has not been introduced
    if 's_crps_loss' not in settings.keys():
        settings['s_crps_loss'] = False

    if settings['s_log_transform']:
        transform_f = lambda x: np.log(x + 1) if isinstance(x, np.ndarray) else torch.log(x + 1)
    else:
        transform_f = lambda x: x

    model = load_from_checkpoint(ps_runs_path, checkpoint_name, linspace_binning_params, settings,
                                 filter_and_normalization_params=filter_and_normalization_params, **plot_settings)
    train_data_loader, validation_data_loader = create_data_loaders(transform_f, filtered_indecies_training, filtered_indecies_validation,
                        linspace_binning_params, filter_and_normalization_params, settings)

    if ps_plot_snapshots:
        checkpoint_name_no_ending = checkpoint_name.replace('.ckpt', {})
        plot_snapshots(model, train_data_loader, filter_and_normalization_params, linspace_binning_params, transform_f, settings,
                       plot_settings, prefix=f'TRAIN_ckpt_{checkpoint_name_no_ending}',
                       **plot_settings)
        plot_snapshots(model, validation_data_loader, filter_and_normalization_params, linspace_binning_params, transform_f, settings,
                       plot_settings, prefix=f'VAL_ckpt_{checkpoint_name_no_ending}',
                       **plot_settings)
    if ps_plot_fss:
        calc_FSS(model, validation_data_loader, filter_and_normalization_params, linspace_binning_params,
                 settings, plot_settings, **plot_settings, **plot_fss_settings)

        plot_fss_by_scales_one_plot(**settings, **plot_fss_settings, num_lines=5)

    if ps_plot_crps:
        # MEMORY INTENSIVE if run with 'crps_load_steps_crps_from_file': False
        # corresponding ram required as by 4 v100 GPU on fair share with steps_n_ens_members: 1024 and 'fss_calc_on_every_n_th_batch': 1,
        calc_CRPS(model, validation_data_loader, filter_and_normalization_params, linspace_binning_params,
                  settings, plot_settings, steps_settings, **plot_settings, **plot_crps_settings)

        plot_crps(**settings, **plot_crps_settings)

    if ps_plot_spread_skill:

        plot_spread_skill(
            model,
            validation_data_loader,
            filter_and_normalization_params,
            linspace_binning_params,
            settings,
            **plot_settings)


if __name__ == '__main__':

    runs_path = '/home/jan/jan/programming/first_CNN_on_Radolan/runs'
    run_name = 'Run_20231108-115128no_gaussian_blurring_with_exp_lr_schedule'

    runs_path = '{}/{}'.format(runs_path, run_name)

    settings, filtered_indecies_training, filtered_indecies_validation, linspace_binning_params, filter_and_normalization_params \
        = load_data_from_run(runs_path, run_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plot_checkpoint_settings ={
        'ps_runs_path': runs_path, #'{}/runs'.format(os.getcwd()),
        'ps_run_name': run_name,
        'ps_device': settings['device'],
        'ps_checkpoint_name': None,  # If none take checkpoint of last epoch
        'ps_inv_normalize': False,
        'ps_gaussian_smoothing_multiple_sigmas': settings['s_gaussian_smoothing_multiple_sigmas'],
        'ps_multiple_sigmas': settings['s_multiple_sigmas'],
        'ps_plot_snapshots': False,
        'ps_plot_fss': False,
        'ps_plot_crps': True,
    }

    plot_crps_settings = {
        'crps_calc_on_every_n_th_batch': 1, #100,
        'crps_load_steps_crps_from_file': False,
        'crps_steps_file_path': None
    }

    steps_settings = {
        'steps_n_ens_members': 16, #1024, # 16, Keep in mind that there are 64 bins!!!
        'steps_num_workers': 16,
    }

    plot_fss_settings = {
        'fss_space_threshold': [0.1, 50, 100], # [1, 20, 20], # start, stop, steps
        'fss_linspace_scale': [1, 10, 100], # start, stop, steps
        'fss_calc_on_every_n_th_batch': 1,
        'fss_log_thresholds': True,
    }


    # Catch previous versions where certain features were not implemented

    plot_from_checkpoint(plot_fss_settings, plot_crps_settings, steps_settings, plot_checkpoint_settings, **plot_checkpoint_settings)
