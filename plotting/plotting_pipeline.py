from plotting.plot_quality_metrics_from_log import plot_qualities_main, plot_qualities_main_several_sigmas, plot_precipitation_diff
from plotting.plot_lr_scheduler import plot_lr_schedule, plot_sigma_schedule

from plotting.calc_and_plot_from_checkpoint import plot_from_checkpoint

from helper.plotting_helper import load_data_from_logs
from plotting.plot_from_log import plot_logged_metrics


def plotting_pipeline(training_steps_per_epoch, model_l, settings,
                      plot_lr_schedule_boo=True, **__):
    '''
    Pipeline for automatic plotting of several figures
    For plot_lr_schedule=True, model_l is required, otherwise None can be passed for model_l
    '''
    s_dirs = settings['s_dirs']

    key_list_train = ['train_mean_loss', 'train_mean_normed_mse']
    key_list_val = ['val_mean_loss', 'val_mean_normed_mse']
    save_name_list = ['loss', 'mse']
    title_list = ['Loss', 'MSE']

    train_df, val_df, base_train_df, base_val_df = load_data_from_logs(**settings)

    for train_key, val_key, save_name, title in zip(key_list_train, key_list_val, save_name_list, title_list):
        plot_logged_metrics(train_key, val_key, save_name, title, train_df, val_df, xlog=False, ylog=True, **settings)


    if settings['s_log_precipitation_difference']:
        pass
        # # This does not yet work with s_gaussian_smoothing_multiple_sigmas
        # try:
        #     plot_precipitation_diff(plot_from_log_settings, **plot_from_log_settings, **settings)
        # except Exception:
        #     print('Precipitation difference plotting encountered error!')

    # Deepcopy lr_scheduler to make sure steps in instance is not messed up
    # lr_scheduler = copy.deepcopy(model_l.lr_scheduler)


    if settings['s_lr_schedule'] and plot_lr_schedule_boo:

        plot_lr_schedule(model_l.lr_scheduler, training_steps_per_epoch, settings['s_max_epochs'],
                         save_name='lr_scheduler', y_label='Learning Rate', title='LR scheduler',
                         ylog=True, **settings)


    plot_checkpoint_settings ={
        'ps_runs_path': s_dirs['save_dir'], #'{}/runs'.format(os.getcwd()),
        'ps_run_name': settings['s_sim_name'],
        'ps_device': settings['device'],
        'ps_checkpoint_name': None,  # If none take checkpoint of last epoch
        'ps_inv_normalize': False,
        'ps_gaussian_smoothing_multiple_sigmas': settings['s_gaussian_smoothing_multiple_sigmas'],
        'ps_multiple_sigmas': settings['s_multiple_sigmas'],
        'ps_plot_snapshots': True,
        'ps_plot_fss': True,
        'ps_plot_crps': True,
        'ps_num_gpus': settings['s_num_gpus'],
    }

    plot_fss_settings = {
        'fss_space_threshold': [0.1, 50, 100], # start, stop, steps
        'fss_linspace_scale': [1, 10, 100], # start, stop, threshold
        'fss_calc_on_every_n_th_batch': 1,
        'fss_log_thresholds': True,
    }

    plot_crps_settings = {
        'crps_calc_on_every_n_th_batch': 10, #1,
        'crps_load_steps_crps_from_file': True,
        # leave away the .pickle.pgx extension
        'crps_steps_file_path': '/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs/Run_20231211-213613_ID_4631443x_entropy_loss_vectorized_CRPS_eval_no_gaussian/logs/crps_steps'
    }

    steps_settings = {
        'steps_n_ens_members': 300,
        'steps_num_workers': 16,
    }

    if settings['s_local_machine_mode']:
        plot_crps_settings['crps_calc_on_every_n_th_batch'] = 100
        # leave away the .pickle.pgx extension
        plot_crps_settings['crps_steps_file_path'] = '/home/jan/Programming/remote/first_CNN_on_radolan_remote/runs/Run_20240123-161505NO_bin_weighting/logs/crps_steps'
        plot_crps_settings['crps_load_steps_crps_from_file'] = True
        steps_settings['steps_n_ens_members'] = 10
        steps_settings['steps_num_workers'] = 16

    plot_from_checkpoint(plot_fss_settings, plot_crps_settings, steps_settings, plot_checkpoint_settings, **plot_checkpoint_settings)

    if settings['s_max_epochs'] > 10:
        plot_from_checkpoint(plot_fss_settings, plot_crps_settings, steps_settings, plot_checkpoint_settings, epoch=10,
                             **plot_checkpoint_settings)
    # except Exception:
    #     warnings.warn('Image plotting encountered error!')



