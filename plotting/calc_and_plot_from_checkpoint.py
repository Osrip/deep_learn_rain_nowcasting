from helper.checkpoint_handling import load_from_checkpoint, create_data_loaders, load_data_from_run
# from helper.helper_functions import load_zipped_pickle
import numpy as np
import torch
import os

from plotting.plot_snapshots import plot_snapshots
from plotting.plot_eval_one_epoch import plot_CRPS
from plotting.plot_eval_one_epoch import plot_FSS


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


def plot_from_checkpoint(settings, plot_settings, ps_runs_path, ps_run_name, ps_checkpoint_name, epoch=None, **__):
    '''
    Loads corresponding epoch
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
    plot_snapshots(model, train_data_loader, filter_and_normalization_params, linspace_binning_params,
                   plot_settings, prefix='TRAIN_epoch_{}'.format(epoch),
                   **plot_settings)
    plot_snapshots(model, validation_data_loader, filter_and_normalization_params, linspace_binning_params,
                   plot_settings, prefix='VAL_epoch_{}'.format(epoch),
                   **plot_settings)

    plot_FSS(model, validation_data_loader, filter_and_normalization_params, linspace_binning_params, settings, plot_settings,
             ps_runs_path, ps_run_name, ps_checkpoint_name, **__)

    # plot_CRPS(model, validation_data_loader, filter_and_normalization_params, linspace_binning_params,
    #           plot_settings, prefix='VAL_epoch_{}'.format(epoch),
    #           **plot_settings)




if __name__ == '__main__':


    # plot_settings = {
    #     'ps_runs_path': '/home/jan/jan/programming/first_CNN_on_Radolan/runs',
    #     'ps_run_name': 'Run_20230602-191416_test_profiler',
    #     'ps_checkpoint_name': 'model_epoch=1_val_loss=3.92.ckpt',
    # }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_settings = {
        'ps_runs_path': '/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs',
        'ps_run_name': 'Run_20230611-212949_ID_3646156_12_months_training_fixed_csv_logging_mlflow_working_1_gpus_several_runs',
        # 'ps_checkpoint_name': 'model_epoch=0049_val_loss=3.95.ckpt',
        # TODO Implement list with epochs to be plotted with -1 being last epoch
        'ps_checkpoint_name': None, # If none take checkpoint of last epoch
        'ps_device': device,
        'ps_inv_normalize': False,
    }


    plot_from_checkpoint(plot_settings, **plot_settings)
