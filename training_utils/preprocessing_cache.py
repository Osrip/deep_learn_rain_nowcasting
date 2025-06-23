import time
import warnings

import numpy as np

from data_pre_processing import create_data_loaders, preprocess_data
from helper import (
    create_save_name_for_data_loader_vars, load_data_loader_vars, save_data_loader_vars,
    save_dict_pickle_csv, save_zipped_pickle, save_tuple_pickle_csv,
    format_duration, invnorm_linspace_binning, inverse_normalize_data
)


def data_loading(
        settings,
        s_force_data_preprocessing,
        **__
):
    '''
    This tries to load data loader vars, if not possible preprocess data
    If structure of data_loader_vars is changed, automatically changes name in _create_save_name_for_data_loader_vars
    (not super reliable, otherwise manually change prefix)
    '''

    try:
        # When loading data loader vars, the file name is checked for whether log transform was used
        if s_force_data_preprocessing:
            warnings.warn('Forced preprocessing of data as s_force_data_preprocessing == True')
            raise FileNotFoundError('Forced preprocessing of data as s_force_data_preprocessing == True')
        print(f"\n LOADING PREPROCESSED DATA \n ...")
        step_start_time = time.time()
        file_name_data_loader_vars = create_save_name_for_data_loader_vars(**settings)
        print(f'Loading data loader vars from file {file_name_data_loader_vars}')
        data_loader_vars = load_data_loader_vars(settings, **settings)

        (
            train_sample_coords, val_sample_coords,
            train_time_keys, val_time_keys, test_time_keys,
            train_oversampling_weights, val_oversampling_weights,
            radolan_statistics_dict,
            linspace_binning_params
        ) = data_loader_vars
        print(f'\n DONE. Took {format_duration(time.time() - step_start_time)} \n')

    except FileNotFoundError:
        print(f"\n PREPROCESSING DATA \n ...")
        step_start_time = time.time()

        (train_sample_coords, val_sample_coords,
        train_time_keys, val_time_keys, test_time_keys,
        train_oversampling_weights, val_oversampling_weights,
        radolan_statistics_dict,
        linspace_binning_params) = preprocess_data(settings, **settings)

        data_loader_vars = (
            train_sample_coords, val_sample_coords,
            train_time_keys, val_time_keys, test_time_keys,
            train_oversampling_weights, val_oversampling_weights,
            radolan_statistics_dict,
            linspace_binning_params
        )

        save_data_loader_vars(data_loader_vars, settings, **settings)

    (
        train_data_loader, validation_data_loader,
        training_steps_per_epoch,
        validation_steps_per_epoch
    ) = create_data_loaders(
        train_sample_coords, val_sample_coords,
        train_oversampling_weights, val_oversampling_weights,
        radolan_statistics_dict,
        settings,
        **settings
    )
    print(f'\n DONE. Took {format_duration(time.time() - step_start_time)} \n')

    # ! This has the same order as input in train_wrapper !
    return (
        train_data_loader, validation_data_loader,
        training_steps_per_epoch, validation_steps_per_epoch,
        train_time_keys, val_time_keys, test_time_keys,
        train_sample_coords, val_sample_coords,
        radolan_statistics_dict,
        linspace_binning_params,
    )


def save_data(
        radolan_statistics_dict,
        train_sample_coords,
        val_sample_coords,
        linspace_binning_params,
        training_steps_per_epoch,
        sigma_schedule_mapping,
        settings,
        s_dirs,
        **__,
):
    save_dict_pickle_csv('{}/radolan_statistics_dict'.format(s_dirs['data_dir']), radolan_statistics_dict)
    save_zipped_pickle('{}/train_sample_coords'.format(s_dirs['data_dir']), train_sample_coords)
    save_zipped_pickle('{}/val_sample_coords'.format(s_dirs['data_dir']), val_sample_coords)

    mean_filtered_log_data = radolan_statistics_dict['mean_filtered_log_data']
    std_filtered_log_data = radolan_statistics_dict['std_filtered_log_data']

    # Save linspace binning params
    save_tuple_pickle_csv(linspace_binning_params, s_dirs['data_dir'], 'linspace_binning_params')
    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params

    linspace_binning_inv_norm, linspace_binning_max_inv_norm = invnorm_linspace_binning(linspace_binning,
                                                                                        linspace_binning_max,
                                                                                        mean_filtered_log_data,
                                                                                        std_filtered_log_data)
    linspace_binning_min_inv_norm = inverse_normalize_data(
        np.array(linspace_binning_min),
        mean_filtered_log_data,
        std_filtered_log_data)
    linspace_binning_params_inv_norm = (linspace_binning_min_inv_norm,
                                        linspace_binning_max_inv_norm,
                                        linspace_binning_inv_norm)
    save_tuple_pickle_csv(
        linspace_binning_params_inv_norm,
        s_dirs['data_dir'],
        'linspace_binning_params_inv_norm')

    save_dict_pickle_csv('{}/settings'.format(s_dirs['data_dir']), settings)

    # Save sigma scheduler and training steps per epoch for s_only_plotting
    save_zipped_pickle('{}/training_steps_per_epoch'.format(s_dirs['data_dir']), training_steps_per_epoch)
    save_zipped_pickle('{}/sigma_schedule_mapping'.format(s_dirs['data_dir']), sigma_schedule_mapping)
