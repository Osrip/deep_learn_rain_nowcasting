from helper.checkpoint_handling import load_from_checkpoint, create_data_loaders, load_data_from_run
from load_data import inverse_normalize_data
from helper.helper_functions import one_hot_to_mm
from plotting.plot_images import plot_target_vs_pred_with_likelihood
# from helper.helper_functions import load_zipped_pickle
import numpy as np
import torch
import os


def plot_images_inner(model, data_loader, filter_and_normalization_params, linspace_binning_params, ps_runs_path,
                      ps_run_name, ps_checkpoint_name, ps_device, prefix='', **__):

    filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized,\
        linspace_binning_max_unnormalized = filter_and_normalization_params

    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params

    inv_norm = lambda x: inverse_normalize_data(x, mean_filtered_data, std_filtered_data, inverse_log=False,
                                                inverse_normalize=True)

    for i, (input_sequence, target_one_hot, target, _) in enumerate(data_loader):
        input_sequence = input_sequence.to(ps_device)
        model = model.to(ps_device)
        pred = model(input_sequence)

        pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)

        if i == 0:
            # !!! Can also be plotted without input sequence by just leaving input_sequence=None !!!
            plot_target_vs_pred_with_likelihood(inv_norm(target), inv_norm(pred_mm), pred,
                                                linspace_binning=inv_norm(linspace_binning),
                                                vmin=inv_norm(linspace_binning_min),
                                                vmax=inv_norm(linspace_binning_max),
                                                save_path_name= '{}/{}/plots/{}_target_vs_pred_likelihood_{}.png'.format(ps_runs_path
                                                                                                        , ps_run_name
                                                                                                        , prefix
                                                                                                        , ps_checkpoint_name),
                                                title='Validation data (log, not normalized)',
                                                input_sequence = input_sequence,
                                                )
        break


def get_checkpoint_name(ps_runs_path, ps_run_name, epoch=None, **__):
    checkpoint_path = '{}/{}/model'.format(ps_runs_path, ps_run_name)
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

    return checkpoint_names[arg_idx]





def plot_images_outer(plot_settings, ps_runs_path, ps_run_name, ps_checkpoint_name, **__):

    if ps_checkpoint_name == None:
        checkpoint_name = get_checkpoint_name(**plot_settings)
    else:
        checkpoint_name = ps_checkpoint_name


    settings, filtered_indecies_training, filtered_indecies_validation, linspace_binning_params, filter_and_normalization_params \
        = load_data_from_run(ps_runs_path, ps_run_name)

    if settings['s_log_transform']:
        transform_f = lambda x: np.log(x + 1)
    else:
        transform_f = lambda x: x

    model = load_from_checkpoint(ps_runs_path, ps_run_name, checkpoint_name, linspace_binning_params, settings)
    train_data_loader, validation_data_loader = create_data_loaders(transform_f, filtered_indecies_training, filtered_indecies_validation,
                        linspace_binning_params, filter_and_normalization_params, settings)
    plot_images_inner(model, train_data_loader, filter_and_normalization_params, linspace_binning_params, prefix='TRAIN',
                      **plot_settings)
    plot_images_inner(model, validation_data_loader, filter_and_normalization_params, linspace_binning_params, prefix='VAL',
                      **plot_settings)


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
    }



    plot_images_outer(plot_settings, **plot_settings)




    #pass