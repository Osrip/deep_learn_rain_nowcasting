import os

from model.model_lightning_wrapper import NetworkL
from helper.helper_functions import load_zipped_pickle
from torch.utils.data import DataLoader
# from load_data import PrecipitationFilteredDataset
# TODO REWRITE THIS TO HANDLE XARRAY STUFF


def load_from_checkpoint(
        save_dir,
        checkpoint_name,

        settings,
        device,
        s_num_gpus,
        s_mode,
        **__):
    '''
    This directly loads the NetworkL class.
    filter_and_normalization_params is needed for crps loss in Network_l
    '''
    checkpoint_path = '{}/model/{}'.format(save_dir, checkpoint_name)

    print("Loading checkpoint '{}'".format(checkpoint_path))

    if s_mode == 'cluster':
        num_gpus = os.environ.get('SLURM_GPUS_ON_NODE')
    else:
        num_gpus = 1

    model = NetworkL.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                          devices=num_gpus)

    model = model.to(device)
    return model


def create_data_loaders_ckpt_plotting(
        transform_f,
        filtered_indecies_training,
        filtered_indecies_validation,
        linspace_binning_params,
        filter_and_normalization_params,
        settings
):
    # TODO !remove this funktion as this is still part of old plotting pipeline!
    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params

    filtered_indecies, mean_filtered_log_data, std_filtered_log_data, _, _, linspace_binning_min_unnormalized,\
        linspace_binning_max_unnormalized = filter_and_normalization_params

    train_data_set = PrecipitationFilteredDataset(filtered_indecies_training, mean_filtered_log_data, std_filtered_log_data,
                                                  linspace_binning_min, linspace_binning_max, linspace_binning,
                                                  transform_f, settings, **settings)

    validation_data_set = PrecipitationFilteredDataset(filtered_indecies_validation, mean_filtered_log_data,
                                                       std_filtered_log_data,
                                                       linspace_binning_min, linspace_binning_max, linspace_binning,
                                                       transform_f, settings, **settings)

    train_data_loader = DataLoader(
        train_data_set,
        batch_size=settings['s_batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=settings['s_num_workers_data_loader'],
        pin_memory=True)

    validation_data_loader = DataLoader(
        validation_data_set,
        batch_size=settings['s_batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=settings['s_num_workers_data_loader'],
        pin_memory=True)

    # Pin Memory: If you load your samples in the Dataset on CPU and would like to push it during training to the GPU, you can speed
    # up the host to device transfer by enabling pin_memory.

    return train_data_loader, validation_data_loader


def load_data_from_run(runs_path, run_name):
    '''
    Loads data from run
    '''

    settings = load_zipped_pickle('{}/data/settings'.format(runs_path))

    filtered_indecies_training = load_zipped_pickle('{}/data/filtered_indecies_training'.format(runs_path))
    filtered_indecies_validation = load_zipped_pickle('{}/data/filtered_indecies_validation'.format(runs_path))

    linspace_binning_params = load_zipped_pickle('{}/data/linspace_binning_params'.format(runs_path))

    filter_and_normalization_params = load_zipped_pickle('{}/data/filter_and_normalization_params'.format(runs_path))

    return settings, filtered_indecies_training, filtered_indecies_validation, linspace_binning_params, filter_and_normalization_params


def get_checkpoint_names(save_dir, **__):
    '''
    Get the filenames of all checkpoints
    '''
    checkpoint_path = '{}/model'.format(save_dir)
    checkpoint_names = []
    for file in os.listdir(checkpoint_path):
        # check only checkpoint files
        if file.endswith('.ckpt'):
            checkpoint_names.append(file)

    return checkpoint_names
