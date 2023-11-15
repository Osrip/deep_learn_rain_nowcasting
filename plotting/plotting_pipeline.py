from plotting.plot_quality_metrics_from_log import plot_qualities_main, plot_qualities_main_several_sigmas, plot_precipitation_diff
from plotting.plot_lr_scheduler import plot_lr_schedule, plot_sigma_schedule

from plotting.calc_and_plot_from_checkpoint import plot_from_checkpoint


def plotting_pipeline(sigma_schedule_mapping, training_steps_per_epoch, s_dirs, settings, model_l, plot_lr_schedule_boo=True,
                      **__):
    '''
    Pipeline for automatic plotting of several figures
    For plot_lr_schedule=True, model_l is required, otherwise None can be passed for model_l
    '''

    plot_metrics_settings = {
        'ps_sim_name': s_dirs['save_dir'] # settings['s_sim_name']
    }

    if not settings['s_gaussian_smoothing_multiple_sigmas']:
        plot_qualities_main(plot_metrics_settings, **plot_metrics_settings, **settings)
    else:
        plot_qualities_main_several_sigmas(plot_metrics_settings, **plot_metrics_settings, **settings)

    if settings['s_log_precipitation_difference']:
        # This does not yet work with s_gaussian_smoothing_multiple_sigmas
        try:
            plot_precipitation_diff(plot_metrics_settings, **plot_metrics_settings, **settings)
        except Exception:
            print('Precipitation difference plotting encountered error!')

    # Deepcopy lr_scheduler to make sure steps in instance is not messed up
    # lr_scheduler = copy.deepcopy(model_l.lr_scheduler)

    plot_lr_schedule_settings = {
        'ps_sim_name': s_dirs['save_dir'] # settings['s_sim_name'], # TODO: Solve conflicting name convention
    }

    if settings['s_lr_schedule'] and plot_lr_schedule_boo:

        plot_lr_schedule(model_l.lr_scheduler, training_steps_per_epoch, settings['s_max_epochs'],
                         save_name='lr_scheduler', y_label='Learning Rate', title='LR scheduler',
                         ylog=True, **plot_lr_schedule_settings)

    if settings['s_schedule_sigma_smoothing']:
        plot_sigma_schedule(sigma_schedule_mapping, save_name='sigma_scheduler', ylog=True, save=True,
                            **plot_lr_schedule_settings)

    # plot_lr_schedule(sigma_scheduler, training_steps_per_epoch, settings['s_max_epochs'],
    #                  init_learning_rate=settings['s_learning_rate'], save_name='sigma_scheduler',
    #                  y_label='Sigma', title='Sigma scheduler', ylog=False, **plot_lr_schedule_settings)

    plot_checkpoint_settings ={
        'ps_runs_path': s_dirs['save_dir'], #'{}/runs'.format(os.getcwd()),
        'ps_run_name': settings['s_sim_name'],
        'ps_device': settings['device'],
        'ps_checkpoint_name': None,  # If none take checkpoint of last epoch
        'ps_inv_normalize': False,
        'ps_gaussian_smoothing_multiple_sigmas': settings['s_gaussian_smoothing_multiple_sigmas'],
        'ps_multiple_sigmas': settings['s_multiple_sigmas'],
    }

    plot_fss_settings = {
        'fss_space_threshold': [0.01, 1, 10], # start, stop, steps
        'fss_linspace_scale': [1, 20, 10], # start, stop, threshold
        'fss_log_thresholds': True,
    }

    plot_from_checkpoint(plot_fss_settings, plot_checkpoint_settings, **plot_checkpoint_settings)

    if settings['s_max_epochs'] > 10:
        plot_from_checkpoint(plot_fss_settings, plot_checkpoint_settings, epoch=10, **plot_checkpoint_settings)
    # except Exception:
    #     warnings.warn('Image plotting encountered error!')



