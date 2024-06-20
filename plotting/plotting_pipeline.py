
from plotting.plot_lr_scheduler import plot_lr_schedule, plot_sigma_schedule

from plotting.calc_and_plot_from_checkpoint import plot_from_checkpoint

from helper.plotting_helper import load_data_from_logs, get_checkpoint_names
from plotting.plot_from_log import plot_logged_metrics, plot_mean_predictions


def plotting_pipeline(training_steps_per_epoch, model_l, settings,
                      plot_lr_schedule_boo=True, **__):
    '''
    Pipeline for automatic plotting of several figures
    For plot_lr_schedule=True, model_l is required, otherwise None can be passed for model_l
    '''
    s_dirs = settings['s_dirs']

    if True:
        ###### Plot logged metrics ######

        key_list_train_mean = ['train_mean_loss', 'train_mean_rmse']
        key_list_val_mean = ['val_mean_loss', 'val_mean_rmse']
        key_list_train_std = ['train_std_loss', 'train_std_rmse']
        key_list_val_std = ['val_std_loss', 'val_std_rmse']
        save_name_list = ['loss', 'mse']
        title_list = ['Loss', 'RMSE (mm)']

        train_df, val_df, base_train_df, base_val_df = load_data_from_logs(**settings)

        for train_mean_key, val_mean_key, train_std_key, val_std_key, save_name, title in\
                zip(key_list_train_mean, key_list_val_mean, key_list_train_std, key_list_val_std, save_name_list, title_list):
            plot_logged_metrics(
                train_df, val_df,
                train_mean_key, val_mean_key, train_std_key, val_std_key,
                save_name, title,
                xlog=False, ylog=True, **settings)

        ###### Plot mean predictions ######

        plot_mean_predictions(train_df, val_df,
                              train_mean_pred_key='train_mean_mean_pred',
                              val_mean_pred_key='val_mean_mean_pred',
                              train_std_pred_key='train_std_mean_pred',
                              val_std_pred_key='val_std_mean_pred',
                              train_mean_target_key='train_mean_mean_target',
                              val_mean_target_key='val_mean_mean_target',
                              train_std_target_key='train_std_mean_target',
                              val_std_target_key='val_std_mean_target',
                              save_name='mean_predictions_targets', title='Mean precipitation (mm)',
                              xlog=False, ylog=True, **settings)

        ###### PLot lr scheduler ######

        if settings['s_lr_schedule'] and plot_lr_schedule_boo:

            plot_lr_schedule(model_l.lr_scheduler, training_steps_per_epoch, settings['s_max_epochs'],
                             save_name='lr_scheduler', y_label='Learning Rate', title='LR scheduler',
                             ylog=True, **settings)

    ###### Plot from checkpoint ######

    plot_checkpoint_settings ={
        'ps_runs_path': s_dirs['save_dir'], #'{}/runs'.format(os.getcwd()),
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
        'fss_space_threshold': [0.1, 50, 100], # start, stop, steps
        'fss_linspace_scale': [1, 10, 100], # start, stop, threshold
        'fss_calc_on_every_n_th_batch': 1,
        'fss_log_thresholds': True,
    }

    plot_crps_settings = {
        'crps_calc_on_every_n_th_batch': 10, #1,
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
        plot_crps_settings['crps_steps_file_path'] =\
            ('/home/jan/Programming/remote/first_CNN_on_radolan_remote/runs'
             '/Run_20240123-161505NO_bin_weighting/logs/crps_steps')
        plot_crps_settings['crps_load_steps_crps_from_file'] = True
        steps_settings['steps_n_ens_members'] = 10
        steps_settings['steps_num_workers'] = 16

    checkpoint_names = get_checkpoint_names(**plot_checkpoint_settings)
    for checkpoint_name in checkpoint_names:
        plot_from_checkpoint(
            checkpoint_name,
            plot_fss_settings,
            plot_crps_settings,
            steps_settings,
            plot_checkpoint_settings,
            **plot_checkpoint_settings)





