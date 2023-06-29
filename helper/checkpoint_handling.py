from train_lightning import Network_l
from helper.helper_functions import load_zipped_pickle
from torch.utils.data import DataLoader
from load_data import PrecipitationFilteredDataset


def load_from_checkpoint(runs_path, run_name, checkpoint_name, linspace_binning_params, settings):
    checkpoint_path = '{}/{}/model/{}'.format(runs_path, run_name, checkpoint_name)
    model = Network_l.load_from_checkpoint(checkpoint_path=checkpoint_path, linspace_binning_params=linspace_binning_params,
                                           **settings)

    return model


def create_data_loaders(transform_f, filtered_indecies_training, filtered_indecies_validation,
                        linspace_binning_params, filter_and_normalization_params, settings):

    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params

    filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized,\
        linspace_binning_max_unnormalized = filter_and_normalization_params

    train_data_set = PrecipitationFilteredDataset(filtered_indecies_training, mean_filtered_data, std_filtered_data,
                                                  linspace_binning_min, linspace_binning_max, linspace_binning,
                                                  transform_f, **settings)

    validation_data_set = PrecipitationFilteredDataset(filtered_indecies_validation, mean_filtered_data,
                                                       std_filtered_data,
                                                       linspace_binning_min, linspace_binning_max, linspace_binning,
                                                       transform_f, **settings)

    train_data_loader = DataLoader(train_data_set, batch_size=settings['s_batch_size'], shuffle=True, drop_last=True,
                                   num_workers=settings['s_num_workers_data_loader'])

    validation_data_loader = DataLoader(validation_data_set, batch_size=settings['s_batch_size'], shuffle=True, drop_last=True,
                                   num_workers=settings['s_num_workers_data_loader'])

    return train_data_loader, validation_data_loader


def load_data_from_run(runs_path, run_name):
    '''
    Loads data from run
    '''

    settings = load_zipped_pickle('{}/{}/data/settings'.format(runs_path, run_name))

    filtered_indecies_training = load_zipped_pickle('{}/{}/data/filtered_indecies_training'.format(runs_path, run_name))
    filtered_indecies_validation = load_zipped_pickle('{}/{}/data/filtered_indecies_validation'.format(runs_path, run_name))

    linspace_binning_params = load_zipped_pickle('{}/{}/data/linspace_binning_params'.format(runs_path, run_name))

    filter_and_normalization_params = load_zipped_pickle('{}/{}/data/filter_and_normalization_params'.format(runs_path, run_name))

    return settings, filtered_indecies_training, filtered_indecies_validation, linspace_binning_params, filter_and_normalization_params
