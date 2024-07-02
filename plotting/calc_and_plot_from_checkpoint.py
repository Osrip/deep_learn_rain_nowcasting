
import numpy as np
import torch
import os

from helper.checkpoint_handling import load_from_checkpoint, create_data_loaders_ckpt_plotting, load_data_from_run
from helper.plotting_helper import get_checkpoint_names
from plotting.plot_snapshots import plot_snapshots
from plotting.calc_plot_CRPS import calc_CRPS, plot_crps
from plotting.calc_plot_FSS import calc_FSS, plot_fss_by_scales, plot_fss_by_threshold,\
    plot_fss_by_threshold_one_plot, plot_fss_by_scales_one_plot
from plotting.plot_spread_skill_ratio import plot_spread_skill
from plotting.calc_plot_FSS_ver2 import calc_FSS_ver2
from helper.pre_process_target_input import inverse_normalize_data


def plot_from_checkpoint_wrapper(settings, s_dirs, **__):
    ###### Plot from checkpoint ######
    plot_checkpoint_settings = {
        'ps_runs_path': s_dirs['save_dir'],  # '{}/runs'.format(os.getcwd()),
        'ps_run_name': settings['s_sim_name'],
        'ps_device': settings['device'],
        'ps_inv_normalize': False,
        'ps_gaussian_smoothing_multiple_sigmas': settings['s_gaussian_smoothing_multiple_sigmas'],
        'ps_multiple_sigmas': settings['s_multiple_sigmas'],
        'ps_plot_snapshots': True,
        'ps_plot_fss': False,
        'ps_plot_crps': False,
        'ps_plot_spread_skill': True,
        'ps_num_gpus': settings['s_num_gpus']
    }

    plot_fss_settings = {
        'fss_space_threshold': [0.1, 50, 100],  # start, stop, steps
        'fss_linspace_scale': [1, 10, 100],  # start, stop, threshold
        'fss_calc_on_every_n_th_batch': 1,
        'fss_log_thresholds': True,
    }

    plot_crps_settings = {
        'crps_calc_on_every_n_th_batch': 10,  # 1,
        'crps_load_steps_crps_from_file': True,
        # leave away the .pickle.pgx extension
        'crps_steps_file_path':
            '/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs/'
            'Run_20231211-213613_ID_4631443x_entropy_loss_vectorized_CRPS_eval_no_gaussian/logs/crps_steps'
    }

    steps_settings = {
        'steps_n_ens_members': 300,
        'steps_num_workers': 16,
    }

    if settings['s_local_machine_mode']:
        plot_crps_settings['crps_calc_on_every_n_th_batch'] = 100
        # leave away the .pickle.pgx extension
        plot_crps_settings['crps_steps_file_path'] = \
            ('/home/jan/Programming/remote/first_CNN_on_radolan_remote/runs'
             '/Run_20240123-161505NO_bin_weighting/logs/crps_steps')
        plot_crps_settings['crps_load_steps_crps_from_file'] = True
        steps_settings['steps_n_ens_members'] = 10
        steps_settings['steps_num_workers'] = 16

    checkpoint_names = get_checkpoint_names(**plot_checkpoint_settings)
    # Execute plotting pipeline for all checkpoints:
    for checkpoint_name in checkpoint_names:
        # Only plot last checkpoint (remove this if all should be plotted)
        if 'last' in checkpoint_name:
            plot_from_checkpoint(
                checkpoint_name,
                plot_fss_settings,
                plot_crps_settings,
                steps_settings,
                plot_checkpoint_settings,
                **plot_checkpoint_settings)


def plot_from_checkpoint(
        checkpoint_name,
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

    model = load_from_checkpoint(ps_runs_path, checkpoint_name, settings, **plot_settings)
    model.freeze()

    train_data_loader, validation_data_loader = create_data_loaders_ckpt_plotting(
        transform_f,
        filtered_indecies_training,
        filtered_indecies_validation,
        linspace_binning_params,
        filter_and_normalization_params,
        settings)

    ###
    # TODO Debugging only, remove this!
    _, mean_filtered_log_data, std_filtered_log_data, _, _, _, _ = filter_and_normalization_params

    num_empty = 0
    num_total = 0
    for i, (input_sequence, target) in enumerate(validation_data_loader):
        if num_total >= 1000:
            break
        target_inv_normalized = inverse_normalize_data(target, mean_filtered_log_data, std_filtered_log_data)
        epsilon = 0.001
        for batch_dim_idx in range(target.shape[0]):
            num_total += 1
            if (target_inv_normalized[batch_dim_idx, :, :] < epsilon).all():
                num_empty += 1

    print(
        f'Validation data loader during trainning loop: {num_empty} targets are empty out of a total of {num_total} targets')

    num_empty = 0
    num_total = 0
    for i, (input_sequence, target) in enumerate(train_data_loader):
        if num_total >= 1000:
            break
        target_inv_normalized = inverse_normalize_data(target, mean_filtered_log_data, std_filtered_log_data)
        epsilon = 0.001
        for batch_dim_idx in range(target.shape[0]):
            num_total += 1
            if (target_inv_normalized[batch_dim_idx, :, :] < epsilon).all():
                num_empty += 1

    print(
        f'train data loader during training loop: {num_empty} targets are empty out of a total of {num_total} targets')

    #####

    checkpoint_name_no_ending = checkpoint_name.replace('.ckpt', '')

    calc_FSS_ver2(
        model,
        train_data_loader,
        filter_and_normalization_params,
        linspace_binning_params,
        checkpoint_name_no_ending,
        settings,
        **plot_settings,
        **settings
    )

    if ps_plot_snapshots:
        plot_snapshots(model, train_data_loader, checkpoint_name_no_ending, filter_and_normalization_params, linspace_binning_params, transform_f, settings,
                       plot_settings, prefix=f'TRAIN_ckpt_{checkpoint_name_no_ending}',
                       **plot_settings)
        plot_snapshots(model, validation_data_loader, checkpoint_name_no_ending, filter_and_normalization_params, linspace_binning_params, transform_f, settings,
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
            checkpoint_name_no_ending,
            settings,
            **plot_settings,
            **settings
        )


if __name__ == '__main__':

    runs_path = '/home/jan/jan/programming/first_CNN_on_Radolan/runs'
    run_name = 'Run_20231108-115128no_gaussian_blurring_with_exp_lr_schedule'

    runs_path = '{}/{}'.format(runs_path, run_name)

    settings, filtered_indecies_training, filtered_indecies_validation, linspace_binning_params, filter_and_normalization_params \
        = load_data_from_run(runs_path, run_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plot_checkpoint_settings ={
        'ps_runs_path': runs_path,  # '{}/runs'.format(os.getcwd()),
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

    plot_from_checkpoint(
        plot_fss_settings,
        plot_crps_settings,
        steps_settings,
        plot_checkpoint_settings,
        **plot_checkpoint_settings)
