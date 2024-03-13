
import numpy as np
import torch
import os



from helper.checkpoint_handling import load_from_checkpoint, create_data_loaders, load_data_from_run
# from helper.helper_functions import load_zipped_pickle
from plotting.plot_snapshots import plot_snapshots
from plotting.calc_plot_CRPS import calc_CRPS, plot_crps
from plotting.calc_plot_FSS import calc_FSS, plot_fss_by_scales, plot_fss_by_threshold,\
    plot_fss_by_threshold_one_plot, plot_fss_by_scales_one_plot



def get_checkpoint_name(ps_runs_path, epoch=None, **__):
    checkpoint_path = '{}/model'.format(ps_runs_path)
    checkpoint_names = []
    for file in os.listdir(checkpoint_path):
        # check only text files
        if file.endswith('.ckpt'):
            checkpoint_names.append(file)

    corresponding_epochs = []
    expression = 'model_epoch='
    for checkpoint_name in checkpoint_names:

        index = checkpoint_name.find(expression)
        index += len(expression)

        epoch_num = checkpoint_name[index:index+4]
        epoch_num = int(epoch_num)
        corresponding_epochs.append(epoch_num)

    if epoch is None:
        arg_idx = np.argmax(corresponding_epochs)

    else:
        arg_idx = corresponding_epochs.index(epoch)

    return checkpoint_names[arg_idx], arg_idx


def plot_from_checkpoint(plot_fss_settings, plot_crps_settings, steps_settings, plot_settings, ps_runs_path, ps_run_name, ps_checkpoint_name, ps_plot_snapshots,
                         ps_plot_fss, ps_plot_crps, ps_num_gpus, epoch=None, **__):
    '''
    Loads model from corresponding epoch and plotsthings up
    This does a forward pass! GPU resources required!
    '''

    if ps_checkpoint_name == None:
        checkpoint_name, epoch = get_checkpoint_name(epoch=epoch, **plot_settings)
    else:
        checkpoint_name, epoch = ps_checkpoint_name


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
        plot_snapshots(model, train_data_loader, filter_and_normalization_params, linspace_binning_params, transform_f, settings,
                       plot_settings, prefix='TRAIN_epoch_{}'.format(epoch),
                       **plot_settings)
        plot_snapshots(model, validation_data_loader, filter_and_normalization_params, linspace_binning_params, transform_f, settings,
                       plot_settings, prefix='VAL_epoch_{}'.format(epoch),
                       **plot_settings)
    if ps_plot_fss:
        calc_FSS(model, validation_data_loader, filter_and_normalization_params, linspace_binning_params,
                 settings, plot_settings, **plot_settings, **plot_fss_settings)
        if False:
            plot_fss_by_scales(**settings, **plot_fss_settings)
            plot_fss_by_threshold(**settings, **plot_fss_settings, num_plots=5)
            plot_fss_by_threshold_one_plot(**settings, **plot_fss_settings, num_lines=5)
        plot_fss_by_scales_one_plot(**settings, **plot_fss_settings, num_lines=5)

    if ps_plot_crps:
        # MEMORY INTENSIVE if run with 'crps_load_steps_crps_from_file': False
        # corresponding ram required as by 4 v100 GPU on fair share with steps_n_ens_members: 1024 and 'fss_calc_on_every_n_th_batch': 1,
        calc_CRPS(model, validation_data_loader, filter_and_normalization_params, linspace_binning_params,
                  settings, plot_settings, steps_settings, **plot_settings, **plot_crps_settings)

        plot_crps(**settings, **plot_crps_settings)

        # print('CRPS model mean: {}'.format(crps_model_mean))
        # print('CRPS model std: {}'.format(crps_model_std))
        # print('CRPS steps mean: {}'.format(crps_steps_mean))
        # print('CRPS steps std: {}'.format(crps_steps_std))



    # plot_CRPS(model, validation_data_loader, filter_and_normalization_params, linspace_binning_params,
    #           plot_settings, prefix='VAL_epoch_{}'.format(epoch),
    #           **plot_settings)


if __name__ == '__main__':

    # # Set wdir to parent dir of plotting:
    #
    # # Get the current working directory (cwd)
    # current_wdir = os.getcwd()
    #
    # # Get the parent directory of the current working directory
    # parent_wdir = os.path.dirname(current_wdir)
    #
    # # Set the parent directory as the current working directory
    # os.chdir(parent_wdir)


    # plot_settings = {
    #     'ps_runs_path': '/home/jan/jan/programming/first_CNN_on_Radolan/runs',
    #     'ps_run_name': 'Run_20230602-191416_test_profiler',
    #     'ps_checkpoint_name': 'model_epoch=1_val_loss=3.92.ckpt',
    # }
    # runs_path = '/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs'
    # run_name = 'Run_20231025-102508_ID_4495294several_seperate_sigmas_01_05_1_2_CONTROL_bernstein_100_epochs_averaged_baseline_NO_lr_scheduler'
    # run_name = 'Run_20231025-143021_ID_4495295several_seperate_sigmas_01_05_1_2_CONTROL_bernstein_100_epochs_averaged_baseline_NO_lr_scheduler'
    # run_name = 'Run_20231207-212109_ID_4624317crps_loss_no_gaussian_blurring'
    # run_name = 'Run_20231024-085515_ID_4492846no_gaussian_100_epochs_averaged_baseline_with_lr_scheduler'
    # run_name = 'Run_20231208-194413_ID_4625720crps_loss_no_gaussian_blurring'
    # run_name = 'Run_20231025-102508_ID_4495294several_seperate_sigmas_01_05_1_2_CONTROL_bernstein_100_epochs_averaged_baseline_NO_lr_scheduler'
    # run_name = 'Run_20231211-213613_ID_4631443x_entropy_loss_vectorized_CRPS_eval_no_gaussian'

    #
    runs_path = '/home/jan/jan/programming/first_CNN_on_Radolan/runs'
    run_name = 'Run_20231108-115128no_gaussian_blurring_with_exp_lr_schedule'

    runs_path = '{}/{}'.format(runs_path, run_name)

    settings, filtered_indecies_training, filtered_indecies_validation, linspace_binning_params, filter_and_normalization_params \
        = load_data_from_run(runs_path, run_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # plot_settings = {
    #     'ps_runs_path': '/home/jan/jan/programming/first_CNN_on_Radolan/runs', # '/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs',
    #     'ps_run_name': 'Run_20231108-115128no_gaussian_blurring_with_exp_lr_schedule', #'Run_20230611-212949_ID_3646156_12_months_training_fixed_csv_logging_mlflow_working_1_gpus_several_runs',
    #     # 'ps_checkpoint_name': 'model_epoch=0049_val_loss=3.95.ckpt',
    #     # TODO Implement list with epochs to be plotted with -1 being last epoch
    #     'ps_checkpoint_name': None, # If none take checkpoint of last epoch
    #     'ps_device': device,
    #     'ps_inv_normalize': False,
    # }

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
        'ps_num_gpus': 4,
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

    # Good qual setings:
    #
    # plot_fss_settings = {
    #     'fss_space_threshold': [0.01, 1, 30], # start, stop, steps
    #     'fss_linspace_scale': [1, 20, 30], # start, stop, steps
    #     'fss_calc_on_every_n_th_batch': 10
    #     'fss_log_thresholds': True,
    # }


    plot_fss_settings = {
        'fss_space_threshold': [0.1, 50, 100], # [1, 20, 20], # start, stop, steps
        'fss_linspace_scale': [1, 10, 100], # start, stop, steps
        'fss_calc_on_every_n_th_batch': 1,
        'fss_log_thresholds': True,
    }



    # Debug settings:

    # plot_fss_settings = {
    #     'fss_space_threshold': [0.1, 10, 50], # start, stop, steps
    #     'fss_linspace_scale': [1, 10, 50], # start, stop, steps
    #     'fss_calc_on_every_n_th_batch': 100,
    #     'fss_log_thresholds': True,
    # }


    # Catch previous versions where certain features were not implemented

    plot_from_checkpoint(plot_fss_settings, plot_crps_settings, steps_settings, plot_checkpoint_settings, **plot_checkpoint_settings)
