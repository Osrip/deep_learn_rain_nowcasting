
import numpy as np
import torch
import os



from helper.checkpoint_handling import load_from_checkpoint, create_data_loaders, load_data_from_run
# from helper.helper_functions import load_zipped_pickle
from plotting.plot_snapshots import plot_snapshots
from plotting.plot_eval_one_epoch import plot_CRPS
from plotting.plot_eval_one_epoch import calc_FSS, plot_fss_by_scales, plot_fss_by_threshold,\
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


def plot_from_checkpoint(plot_fss_settings, plot_settings, ps_runs_path, ps_run_name, ps_checkpoint_name, ps_plot_snapshots,
                         ps_plot_fss, epoch=None, **__):
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

    if settings['s_log_transform']:
        transform_f = lambda x: np.log(x + 1)
    else:
        transform_f = lambda x: x

    model = load_from_checkpoint(ps_runs_path, ps_run_name, checkpoint_name, linspace_binning_params, settings)
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

        plot_fss_by_scales(**settings)
        plot_fss_by_threshold(**settings,  num_plots=5)
        plot_fss_by_threshold_one_plot(**settings, num_lines=5)
        plot_fss_by_scales_one_plot(**settings, num_lines=5)

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
    runs_path = '/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs'
    run_name = 'Run_20231025-102508_ID_4495294several_seperate_sigmas_01_05_1_2_CONTROL_bernstein_100_epochs_averaged_baseline_NO_lr_scheduler'
    #
    # runs_path = '/home/jan/jan/programming/first_CNN_on_Radolan/runs'
    # run_name = 'Run_20231108-115128no_gaussian_blurring_with_exp_lr_schedule'

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
        'ps_plot_snapshots': True,
        'ps_plot_fss': True,
    }

    # Good qual setings:
    #
    # plot_fss_settings = {
    #     'fss_space_threshold': [0.01, 1, 30], # start, stop, steps
    #     'fss_linspace_scale': [1, 20, 30], # start, stop, steps
    #     'fss_calc_on_every_n_th_batch': 10
    # }


    plot_fss_settings = {
        'fss_space_threshold': [1, 20, 20], # start, stop, steps
        'fss_linspace_scale': [1, 10, 20], # start, stop, steps
        'fss_calc_on_every_n_th_batch': 10
    }

    # Debug settings:

    # plot_fss_settings = {
    #     'fss_space_threshold': [0.01, 1, 5], # start, stop, steps
    #     'fss_linspace_scale': [1, 20, 5], # start, stop, threshold
    #     'fss_calc_on_every_n_th_batch': 100
    # }
    #
    # plot_checkpoint_settings['ps_plot_snapshots'] = False


    plot_from_checkpoint(plot_fss_settings, plot_checkpoint_settings, **plot_checkpoint_settings)
