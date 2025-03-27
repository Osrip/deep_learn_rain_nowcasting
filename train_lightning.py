import os
import torch
from network_lightning import NetworkL
import datetime
import time
import xarray as xr
import argparse

from load_data_xarray import (
    create_patches,
    filter_patches,
    calc_statistics_on_valid_patches,
    patch_indices_to_sample_coords,
    calc_linspace_binning,
    FilteredDatasetXr,
    create_split_time_keys,
    split_data_from_time_keys,
    patches_boo_to_datetime_idx_permuts,
    calc_bin_frequencies,
    create_oversampling_weights,
)
from helper.pre_process_target_input import invnorm_linspace_binning, inverse_normalize_data
from helper.memory_logging import format_duration, format_ram_usage
from helper.settings_config_helper import load_settings, load_yaml_config
from helper.helper_functions import (
    save_zipped_pickle,
    save_dict_pickle_csv,
    save_tuple_pickle_csv,
    save_project_code,
    load_zipped_pickle,
    save_data_loader_vars,
    load_data_loader_vars
)

from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np


import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from logger import (ValidationLogsCallback,
                    TrainingLogsCallback,
                    BaselineTrainingLogsCallback,
                    BaselineValidationLogsCallback,
                    create_loggers)
from baselines import LKBaseline
from plotting.plotting_pipeline import plot_logs_pipeline
from helper.sigma_scheduler_helper import create_scheduler_mapping
from helper.helper_functions import no_special_characters, create_save_name_for_data_loader_vars

import warnings
from tests.test_basic_functions import test_all
from pytorch_lightning.loggers import WandbLogger
from evaluation.evaluation_pipeline import evaluation_pipeline


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


def preprocess_data(
        settings,
        s_target_height_width,
        s_input_height_width,
        s_input_padding,
        s_data_variable_name,
        s_split_chunk_duration,
        s_folder_path,
        s_data_file_name,
        s_time_span_for_bin_frequencies,
        **__,
):
    '''
    Patches refer to targets
    Samples refer to the data delivered by data loader
    '''
    print('Preprocess Radolan data:')
    start_time = time.time()

    # Define constants for pre-processing
    y_target, x_target = s_target_height_width, s_target_height_width  # 73, 137 # how many pixels in y and x direction
    y_input, x_input = s_input_height_width, s_input_height_width
    y_input_padding, x_input_padding = s_input_padding, s_input_padding  # Additional padding that the frames that will be returned to data loader get for Augmentation

    # --- PATCH AND FILTER ---
    print(f"Create patches ... {format_ram_usage()}")
    step_start_time = time.time()
    # Patch data into patches of the target size and filter these patches. Patches that passed filter are called 'valid'

    (
        patches,
        # patches: xr.Dataset Patch dimensions y_outer, x_outer give one coordinate pair for each patch,
        # y_inner, x_inner give pixel dimensions for each patch
        data,
        # data: The unpatched data that has global pixel coordinates,
        data_shortened,
        # data_shortened: same as data, but beginning is missing (lead_time + num input frames) such that we can go
        # 'back in time' to go frame target time to input time.
     ) = create_patches(
        y_target, x_target,
        **settings
    )
    print(f"Done. Took {format_duration(time.time() - step_start_time)}")

    # Filter patches
    print(f"Filter patches ... {format_ram_usage()}"); step_start_time = time.time()
    valid_patches_boo = filter_patches(patches, **settings)
    # valid_patches_boo: Boolean xr.Dataset with y_outer and x_outer defines the valid patches
    print(f"Done. Took {format_duration(time.time() - step_start_time)}")

    # --- SPLIT DATA ---
    # We are grouping the data (i.e. daily) and then are splitting these (daily) groups into train, val and test set

    # Resample shortened data, from which the time_keys are generated that determine the splits
    # Each time_key represents one 'time group'
    # The splits are created on the time stamps of the targets, which the patches are linked to.
    print(f"Split data ... {format_ram_usage()}"); step_start_time = time.time()
    resampled_data = data_shortened.resample(time=s_split_chunk_duration)
    # Randomly split the time_keys
    train_time_keys, val_time_keys, test_time_keys = create_split_time_keys(resampled_data, **settings)

    # Resample the valid_patches_boo into time groups
    resampled_valid_patches_boo = valid_patches_boo.resample(time=s_split_chunk_duration)

    # Split valid_patches_boo into train and val
    train_valid_patches_boo = split_data_from_time_keys(resampled_valid_patches_boo, train_time_keys)
    val_valid_patches_boo = split_data_from_time_keys(resampled_valid_patches_boo, val_time_keys)
    print(f"Done. Took {format_duration(time.time() - step_start_time)}")

    # --- CALC NORMALIZATION STATISTICS on valid training patches---
    # Only calculating on training data to prevent data leakage
    print(f"Calculate normalization statistics on training data ... {format_ram_usage()}"); step_start_time = time.time()
    _, _, mean_filtered_log_data, std_filtered_log_data = calc_statistics_on_valid_patches(
        patches,
        train_valid_patches_boo,
        **settings
    )

    radolan_statistics_dict = {
        'mean_filtered_log_data': mean_filtered_log_data,
        'std_filtered_log_data': std_filtered_log_data
    }
    print(f"Done. Took {format_duration(time.time() - step_start_time)}")

    # --- CREATE LINSPACE BINNING ---
    print(f"Create linspace binning ... {format_ram_usage()}"); step_start_time = time.time()
    linspace_binning_min_normed, linspace_binning_max_normed, linspace_binning_normed = calc_linspace_binning(
        data,
        mean_filtered_log_data,
        std_filtered_log_data,
        **settings,
    )
    linspace_binning_params = linspace_binning_min_normed, linspace_binning_max_normed, linspace_binning_normed
    print(f"Done. Took {format_duration(time.time() - step_start_time)}")

    # --- CALC BIN FREQUENCIES FOR OVERSAMPLING ---
    print(f"Calculate bin frequencies for oversampling ... {format_ram_usage()}"); step_start_time = time.time()

    # Load specific time span to calculate bin frequencies on - quick & dirty
    # as this calculation is extremely expensive

    load_path = '{}/{}'.format(s_folder_path, s_data_file_name)
    data_set = xr.open_dataset(load_path, engine='zarr', chunks=None)
    crop_start, crop_end = np.datetime64(s_time_span_for_bin_frequencies[0]), np.datetime64(s_time_span_for_bin_frequencies[1])
    crop_slice = slice(crop_start, crop_end)

    data_subsampled = data_set.sel(time=crop_slice)

    bin_frequencies = calc_bin_frequencies(
        data_subsampled,
        linspace_binning_params,
        mean_filtered_log_data, std_filtered_log_data,
        **settings,
    )
    print(f"Done. Took {format_duration(time.time() - step_start_time)}")

    # --- INDEX CONVERSION from patch to sample ---
    print(f"Convert patch indices to sample coordinates ... {format_ram_usage()}"); step_start_time = time.time()
    #  outer coordinates (define patches in 'patches')
    # 1. -> outer indecies (define patches in 'patches')
    # 2. -> global sample coordinates (reshaped to input size + augmentation padding)

    # 1. We convert the outer coordinates that define the valid patches to indecies with respect to 'patches'
    # !The spacial and time indecies refer to data_shortened!
    # -> valid_target_indecies_outer contains [[time_datetime, y_outer_idx, x_outer_idx],...] permutations of all valid
    # patches with respect to the 'patches' dataset.

    # We use time_datetime instead of time_idx, as the data has already been split, and we thus cannot calculate
    # in time idx space
    # valid_datetime_idx_permuts: [[time: np.datetime64, y_idx (outer patch dim): int, x_idx (outer patch dim): int], ...]

    train_valid_datetime_idx_permuts = patches_boo_to_datetime_idx_permuts(train_valid_patches_boo, **settings)
    val_valid_datetime_idx_permuts = patches_boo_to_datetime_idx_permuts(val_valid_patches_boo, **settings)

    # --- Check for duplicates ---
    # Check if there are any duplicates in the indices (list of tuples)
    train_set = set(train_valid_datetime_idx_permuts)
    val_set = set(val_valid_datetime_idx_permuts)

    # Find any common elements (duplicates) between the two sets
    duplicates = train_set.intersection(val_set)

    # Raise an error if there are duplicates in train and val
    if len(duplicates) > 0:
        raise ValueError(
            f'There are {len(duplicates)} duplicates in the split indices that train and val data is created from')

    # 2. We scale up the patches from target size to input + augmentation size (which is why we need the pixel indecies
    # created in 1.) and return the sample coordiantes together with the time coordinate of the target frame for the sample
    # -> patch_indecies_to_sample_coords takes all these indecies and converts them to the slices that are needed to
    # cut out the patches from data_shortened.
    # sample coords: [[np.datetime64 of target frame, y slice (coordinates), x slice (coordinates)],...]

    # !This drops samples, that exceed bounds ! Therefore updating valid_datetime_idx_permuts as well
    train_sample_coords, train_valid_datetime_idx_permuts = patch_indices_to_sample_coords(
        data_shortened,
        train_valid_datetime_idx_permuts,
        y_target, x_target,
        y_input, x_input,
        y_input_padding, x_input_padding,
    )

    val_sample_coords, val_valid_datetime_idx_permuts = patch_indices_to_sample_coords(
        data_shortened,
        val_valid_datetime_idx_permuts,
        y_target, x_target,
        y_input, x_input,
        y_input_padding, x_input_padding,
    )
    print(f"Done. Took {format_duration(time.time() - step_start_time)}")

    # --- CREATE OVERSAMPLING ---
    print(f"Create oversampling weights ... {format_ram_usage()}"); step_start_time = time.time()
    # THIS USES NUMPY! NOT OPTIMIZED FOR CHUNKING!
    train_oversampling_weights, val_oversampling_weights = create_oversampling_weights(
        # Using the updated version of valid_datetime_idx_permuts, where the samples that have been dropped in previous
        # step have been removed
        (train_valid_datetime_idx_permuts, val_valid_datetime_idx_permuts),
        patches,
        bin_frequencies,
        linspace_binning_params,
        mean_filtered_log_data,
        std_filtered_log_data,
        **settings
    )
    print(f"Done. Took {format_duration(time.time() - step_start_time)}")
    print(f"Preprocessing complete. Total time: {format_duration(time.time() - start_time)}")

    return (
        train_sample_coords, val_sample_coords,
        train_time_keys, val_time_keys, test_time_keys,
        train_oversampling_weights, val_oversampling_weights,
        radolan_statistics_dict,
        linspace_binning_params,
    )


def create_data_loaders(
        train_sample_coords, val_sample_coords,
        train_oversampling_weights, val_oversampling_weights,
        radolan_statistics_dict,

        settings,
        s_batch_size,
        s_num_workers_data_loader,
        s_oversample_validation,
        s_oversample_train,
        s_train_samples_per_epoch,
        s_val_samples_per_epoch,
        **__
):
    '''
    Creates data loaders for training and validation data
    '''

    train_data_set = FilteredDatasetXr(
        train_sample_coords,
        radolan_statistics_dict,
        mode='train',
        settings=settings,
    )

    val_data_set = FilteredDatasetXr(
        val_sample_coords,
        radolan_statistics_dict,
        mode='train',
        settings=settings,
    )

    if s_train_samples_per_epoch is not None:
        train_samples_per_epoch = s_train_samples_per_epoch
    else:
        train_samples_per_epoch = len(train_data_set)

    if s_val_samples_per_epoch is not None:
        val_samples_per_epoch = s_val_samples_per_epoch
    else:
        val_samples_per_epoch = len(val_data_set)

    # TODO: Try Log weights instead (--> apple note 'Bin Frequencies for Oversampling in xarray')

    train_weighted_random_sampler = WeightedRandomSampler(weights=np.power(train_oversampling_weights, 0.5),
                                                          num_samples=train_samples_per_epoch,
                                                          replacement=True)

    val_weighted_random_sampler = WeightedRandomSampler(weights=np.power(val_oversampling_weights, 0.5), #Taking sqrt
                                                        num_samples=val_samples_per_epoch,
                                                        replacement=True)

    if not len(train_oversampling_weights) == len(train_data_set) == len(train_sample_coords):
        raise ValueError('Length of oversampling weights does not match length of data set or sample coords')

    if not len(val_oversampling_weights) == len(val_data_set) == len(val_sample_coords):
        raise ValueError('Length of oversampling weights does not match length of data set or sample coords')

    if s_oversample_train:
        # ... with oversampling:
        train_data_loader = DataLoader(
            train_data_set,
            sampler=train_weighted_random_sampler,  # <-- OVERSAMPLING
            batch_size=s_batch_size,
            drop_last=False,
            num_workers=s_num_workers_data_loader,
            pin_memory=True
        )
    else:
        # ... without oversampling:
        train_data_loader = DataLoader(
            train_data_set,
            batch_size=s_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=s_num_workers_data_loader,
            pin_memory=True
        )

    # Validation data loader
    if s_oversample_validation:
        # ... with oversampling:
        validation_data_loader = DataLoader(
            val_data_set,
            sampler=val_weighted_random_sampler, # <-- OVERSAMPLING
            batch_size=s_batch_size,
            drop_last=False,
            num_workers=s_num_workers_data_loader,
            pin_memory=True)
    else:
        # ... without oversampling:
        validation_data_loader = DataLoader(
            val_data_set,
            batch_size=s_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=s_num_workers_data_loader,
            pin_memory=True)



    return train_data_loader, validation_data_loader, train_samples_per_epoch, val_samples_per_epoch


def calc_baselines(data_loader_list, logs_callback_list, logger_list, logging_type_list, mean_filtered_log_data_list,
                   std_filtered_log_data_list, settings, s_epoch_repetitions_baseline, **__):
    '''
    Goes into train_wrapper
    data_loader_list, logs_callback_list, logger_list, logging_type_list have to be in according order
    logging_type depending on data loader either: 'train' or 'val'
    '''

    for data_loader, logs_callback, logger, logging_type, mean_filtered_log_data, std_filtered_log_data in \
            zip(data_loader_list, logs_callback_list, logger_list, logging_type_list, mean_filtered_log_data_list,
                std_filtered_log_data_list):
        # Create callback list in the form of [BaselineTrainingLogsCallback(base_train_logger)]
        callback_list_base = [logs_callback(logger)]

        lk_baseline = LKBaseline(logging_type, mean_filtered_log_data, std_filtered_log_data, **settings)

        trainer = pl.Trainer(callbacks=callback_list_base, max_epochs=s_epoch_repetitions_baseline, log_every_n_steps=1,
                             check_val_every_n_epoch=1)
        trainer.validate(lk_baseline, data_loader)


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


def train_wrapper(
        train_data_loader, validation_data_loader,
        training_steps_per_epoch, validation_steps_per_epoch,
        train_time_keys, val_time_keys, test_time_keys,
        train_sample_coords, val_sample_coords,
        radolan_statistics_dict,
        linspace_binning_params,

        settings,
        s_dirs, s_profiling, s_max_epochs, s_sim_name,
        s_gaussian_smoothing_target, s_sigma_target_smoothing, s_schedule_sigma_smoothing,
        s_train_samples_per_epoch, s_val_samples_per_epoch,
        s_calc_baseline,
        s_batch_size,
        s_mode,
        **__
):
    """
    All the junk surrounding train_l() goes in here
    Please keep intput arguments in the same order as the output of create_data_loaders()
    """

    print(f"\nTRAINING DATA LOADER:\n"
          f"  Num samples: {len(train_data_loader.dataset)} "
          f"(Num batches: {len(train_data_loader)})\n"
          f"  Samples per epoch: {s_train_samples_per_epoch if s_train_samples_per_epoch is not None else '= Num samples'}\n"
          f"\nVALIDATION DATA LOADER:\n"
          f"  Num samples: {len(validation_data_loader.dataset)} "
          f"(Num batches: {len(validation_data_loader)})\n"
          f"  Samples per epoch: {s_val_samples_per_epoch if s_val_samples_per_epoch is not None else '= Num samples'}\n")

    print(f"Batch size: {train_data_loader.batch_size}" +
          (f" (different for validation: {validation_data_loader.batch_size})"
           if train_data_loader.batch_size != validation_data_loader.batch_size else ""))

    train_logger, val_logger, base_train_logger, base_val_logger = create_loggers(**settings)

    # This is used to save checkpoints of the model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=s_dirs['model_dir'],
        monitor='val_mean_loss',
        filename='model_epoch_{epoch:04d}_valmeanloss_{val_mean_loss:.2f}_best',
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,  # Prevent special characters in file name
    )
    # save_top_k=-1, prevents callback from overwriting previous checkpoints

    if s_profiling:
        profiler = PyTorchProfiler(dirpath=s_dirs['profile_dir'], export_to_chrome=True)
    else:
        profiler = None

    # Increase time out for weights and biases to prevent time out on galavani
    os.environ["WANDB__SERVICE_WAIT"] = "600"

    # Different project names depending on mode:

    wandb_project_name = s_mode


    logger = WandbLogger(name=s_sim_name, project=wandb_project_name)
    # logger = None

    callback_list = [
        checkpoint_callback,
        TrainingLogsCallback(train_logger),
        ValidationLogsCallback(val_logger)
    ]

    if s_gaussian_smoothing_target and s_schedule_sigma_smoothing:

        sigma_schedule_mapping, sigma_scheduler = create_scheduler_mapping(training_steps_per_epoch, s_max_epochs,
                                                                           s_sigma_target_smoothing, **settings)
    else:
        sigma_schedule_mapping, sigma_scheduler = (None, None)

    save_data(
        radolan_statistics_dict,
        train_sample_coords,
        val_sample_coords,
        linspace_binning_params,
        training_steps_per_epoch,
        sigma_schedule_mapping,
        settings,
        s_dirs,
        **__,
    )

    save_project_code(s_dirs['code_dir'])

    print(f"\n STARTING TRAINING \n ...")
    step_start_time = time.time()

    model_l = train_l(
        train_data_loader, validation_data_loader,
        profiler,
        callback_list,
        logger,
        training_steps_per_epoch,
        radolan_statistics_dict,
        linspace_binning_params,
        sigma_schedule_mapping,
        settings,
        **settings)

    print(f'\n DONE. Took {format_duration(time.time() - step_start_time)} \n')

    if s_calc_baseline:
        calc_baselines(**settings,
                       data_loader_list=[train_data_loader, validation_data_loader],
                       logs_callback_list=[BaselineTrainingLogsCallback, BaselineValidationLogsCallback],
                       logger_list=[base_train_logger, base_val_logger],
                       logging_type_list=['train', 'val'],
                       mean_filtered_log_data_list=[radolan_statistics_dict['mean_filtered_log_data'],
                                                radolan_statistics_dict['mean_filtered_log_data']],
                       std_filtered_log_data_list=[radolan_statistics_dict['std_filtered_log_data'],
                                                  radolan_statistics_dict['std_filtered_log_data']],
                       settings=settings
                       )

    # Network_l, training_steps_per_epoch is returned to be able to plot lr_scheduler
    return model_l, training_steps_per_epoch, sigma_schedule_mapping


def train_l(
        train_data_loader, validation_data_loader,
        profiler,
        callback_list,
        logger,
        training_steps_per_epoch,
        radolan_statistics_dict,
        linspace_binning_params,
        sigma_schedule_mapping,

        settings,
        s_max_epochs,
        s_num_gpus,
        s_check_val_every_n_epoch,
        **__):
    '''
    Train loop, keep this clean!
    '''

    # load static and dynamic statistics dicts here from train data loader
    # and pass them to Network_l
    dynamic_statistics_dict_train_data = train_data_loader.dataset.dynamic_statistics_dict
    static_statistics_dict_train_data = train_data_loader.dataset.static_statistics_dict

    # Statistics are only extracted from training data to prevent data leakage and ensure
    # consistency for model learning.

    model_l = NetworkL(
        dynamic_statistics_dict_train_data,
        static_statistics_dict_train_data,
        linspace_binning_params,
        sigma_schedule_mapping,
        settings,
        training_steps_per_epoch=training_steps_per_epoch,
        **settings)

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callback_list,
        profiler=profiler,
        max_epochs=s_max_epochs,
        log_every_n_steps=100, # 100 TODO Does this lead to slow-down?
        logger=logger,
        devices='auto',
        check_val_every_n_epoch=s_check_val_every_n_epoch,
        strategy='ddp',
        num_sanity_val_steps=0
    )
    # num_sanity_val_steps=0 turns off validation sanity checking
     # precision='16-mixed'
    # 'devices' argument is ignored when device == 'cpu'
    # Speed up advice: https://pytorch-lightning.readthedocs.io/en/1.8.6/guides/speed.html

    # Doing one validation epoch on the untrained model
    trainer.validate(model_l, dataloaders=validation_data_loader)
    # trainer.logger = logger
    trainer.fit(model_l, train_data_loader, validation_data_loader)

    # Network_l instance is returned to be able to plot lr_scheduler
    return model_l


def create_s_dirs(sim_name, s_mode):
    s_dirs = {}
    if s_mode in ['local', 'debug']:
        s_dirs['save_dir'] = f'runs/{s_mode}/{sim_name}'
        s_dirs['prediction_dir'] = f'/home/jan/Programming/weather_data/predictions/{sim_name}'
    else:  # cluster mode
        s_dirs['save_dir'] = f'/home/butz/bst981/nowcasting_project/results/{sim_name}'
        s_dirs['prediction_dir'] = f'/home/butz/bst981/nowcasting_project/output/predictions/{sim_name}'

    # s_dirs['save_dir'] = 'runs/{}'.format(s_sim_name)
    s_dirs['plot_dir']          = '{}/plots'.format(s_dirs['save_dir'])
    s_dirs['plot_dir_images']   = '{}/images'.format(s_dirs['plot_dir'])
    s_dirs['plot_dir_fss']      = '{}/fss'.format(s_dirs['plot_dir'])
    s_dirs['model_dir']         = '{}/model'.format(s_dirs['save_dir'])
    s_dirs['code_dir']          = '{}/code'.format(s_dirs['save_dir'])
    s_dirs['profile_dir']       = '{}/profile'.format(s_dirs['save_dir'])
    s_dirs['logs']              = '{}/logs'.format(s_dirs['save_dir'])
    s_dirs['data_dir']          = '{}/data'.format(s_dirs['save_dir'])
    s_dirs['batches_outputs']   = '{}/batches_outputs'.format(s_dirs['save_dir'])

    return s_dirs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cluster',
                        choices=['cluster', 'local', 'debug'],
                        help="Mode: cluster, local, or debug")
    args = parser.parse_args()

    # Load settings based on mode
    settings = load_settings(args.mode)

    # Process simulation name suffix
    s_sim_name_suffix = settings.get('s_sim_name_suffix', 'dlbd_training_one_month')
    s_sim_name_suffix = no_special_characters(s_sim_name_suffix)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build simulation name based on mode.
    if args.mode in ['local', 'debug']:
        s_sim_name = 'Run_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        s_sim_name = 'Run_{}_ID_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                           int(os.environ.get('SLURM_JOB_ID', 0)))
    s_sim_name += s_sim_name_suffix

    # Create directories based on mode.
    if args.mode in ['local', 'debug']:
        s_dirs = {
            'save_dir': f'runs/{s_sim_name}',
            'prediction_dir': f'/home/jan/Programming/weather_data/predictions/{s_sim_name}'
        }
    else:
        s_dirs = {
            'save_dir': f'/home/butz/bst981/nowcasting_project/results/{s_sim_name}',
            'prediction_dir': f'/home/butz/bst981/nowcasting_project/output/predictions/{s_sim_name}'
        }
    # Append additional directory entries.
    s_dirs['plot_dir'] = f"{s_dirs['save_dir']}/plots"
    s_dirs['plot_dir_images'] = f"{s_dirs['plot_dir']}/images"
    s_dirs['plot_dir_fss'] = f"{s_dirs['plot_dir']}/fss"
    s_dirs['model_dir'] = f"{s_dirs['save_dir']}/model"
    s_dirs['code_dir'] = f"{s_dirs['save_dir']}/code"
    s_dirs['profile_dir'] = f"{s_dirs['save_dir']}/profile"
    s_dirs['logs'] = f"{s_dirs['save_dir']}/logs"
    s_dirs['data_dir'] = f"{s_dirs['save_dir']}/data"
    s_dirs['batches_outputs'] = f"{s_dirs['save_dir']}/batches_outputs"

    # Append extra keys to settings.
    settings['s_mode'] = args.mode
    settings['s_dirs'] = s_dirs
    settings['device'] = device
    settings['s_sim_name'] = s_sim_name


    if not settings['s_plotting_only']:
        for _, make_dir in s_dirs.items():
            if not os.path.exists(make_dir):
                os.makedirs(make_dir)

    if settings['s_no_plotting']:
        for en in ['s_plot_average_preds_boo', 's_plot_pixelwise_preds_boo', 's_plot_target_vs_pred_boo',
                   's_plot_mse_boo', 's_plot_losses_boo', 's_plot_img_histogram_boo']:
            settings[en] = False

    if settings['s_testing']:
        test_all()

    if not settings['s_plotting_only']:
        # --- Normal training ---
        data_set_vars = data_loading(settings, **settings)

        (train_data_loader, validation_data_loader,
        training_steps_per_epoch, validation_steps_per_epoch,
        train_time_keys, val_time_keys, test_time_keys,
        train_sample_coords, val_sample_coords,
        radolan_statistics_dict,
        linspace_binning_params,) = data_set_vars

        model_l, training_steps_per_epoch, sigma_schedule_mapping = train_wrapper(
            *data_set_vars,
            settings,
            **settings
        )

        plot_logs_pipeline(
            training_steps_per_epoch,
            model_l,
            settings, **settings
        )

        evaluation_pipeline(data_set_vars, settings)

    else:
        # --- Plotting only ---
        load_dirs = create_s_dirs(settings['s_plot_sim_name'], settings['s_mode'])
        training_steps_per_epoch = load_zipped_pickle('{}/training_steps_per_epoch'.format(load_dirs['data_dir']))
        sigma_schedule_mapping = load_zipped_pickle('{}/sigma_schedule_mapping'.format(load_dirs['data_dir']))
        ckpt_settings = load_zipped_pickle('{}/settings'.format(load_dirs['data_dir']))

        # Convert some of the loaded settings to the current settings
        ckpt_settings['s_num_gpus']                 = settings['s_num_gpus']
        ckpt_settings['s_baseline_path']            = settings['s_baseline_path']
        ckpt_settings['s_baseline_variable_name']   = settings['s_baseline_variable_name']
        ckpt_settings['s_num_input_frames_baseline']= settings['s_num_input_frames_baseline']

        # Pass settings of the loaded run to get the according data_set_vars
        data_set_vars = data_loading(ckpt_settings, **ckpt_settings)

        evaluation_pipeline(data_set_vars, ckpt_settings)
















