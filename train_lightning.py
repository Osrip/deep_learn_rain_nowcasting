import os
import torch
from network_lightning import NetworkL
import datetime

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
from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np
from helper.helper_functions import save_zipped_pickle, save_dict_pickle_csv,\
    save_tuple_pickle_csv, save_project_code, load_zipped_pickle, save_data_loader_vars, load_data_loader_vars

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

    except FileNotFoundError:
        print('Data loader vars not found, preprocessing data!')

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
        **__,
):
    '''
    Patches refer to targets
    Samples refer to the data delivered by data loader
    '''
    # Define constants for pre-processing
    y_target, x_target = s_target_height_width, s_target_height_width  # 73, 137 # how many pixels in y and x direction
    y_input, x_input = s_input_height_width, s_input_height_width
    y_input_padding, x_input_padding = s_input_padding, s_input_padding  # Additional padding that the frames that will be returned to data loader get for Augmentation

    # --- PATCH AND FILTER ---
    # Patch data into patches of the target size and filter these patches. Patches that passed filter are called 'valid'

    (
        patches,
        # patches: xr.Dataset Patch dimensions y_outer, x_outer give one coordinate pair for each patch,
        # y_inner, x_inner give pixel dimensions for each patch
        data,
        # data: The unpatched data that has global pixel coordinates,
        data_shortened,
        # data_shortened: same as data, but beginning is missing (lead_time + num input frames) such that we can go
        # 'back in time' to go fram target time to input time.
     ) = create_patches(
        y_target, x_target,
        **settings
    )

    # Filter patches
    valid_patches_boo = filter_patches(patches, **settings)
    # valid_patches_boo: Boolean xr.Dataset with y_outer and x_outer defines the valid patches

    # --- SPLIT DATA ---
    # We are grouping the data (i.e. daily) and then are splitting these (daily) groups into train, val and test set

    # Resample shortened data, from which the time_keys are generated that determine the splits
    # Each time_key represents one 'time group'
    # The splits are created on the time stamps of the targets, which the patches are linked to.

    resampled_data = data_shortened.resample(time=s_split_chunk_duration)
    # Randomly split the time_keys
    train_time_keys, val_time_keys, test_time_keys = create_split_time_keys(resampled_data, **settings)

    # Resample the valid_patches_boo into time groups
    resampled_valid_patches_boo = valid_patches_boo.resample(time=s_split_chunk_duration)

    # Split valid_patches_boo into train and val
    train_valid_patches_boo = split_data_from_time_keys(resampled_valid_patches_boo, train_time_keys)
    val_valid_patches_boo = split_data_from_time_keys(resampled_valid_patches_boo, val_time_keys)

    # --- CALC NORMALIZATION STATISTICS on valid training patches---
    # Only calculating on training data to prevent data leakage
    _, _, mean_filtered_log_data, std_filtered_log_data = calc_statistics_on_valid_patches(
        patches,
        train_valid_patches_boo,
        **settings
    )

    radolan_statistics_dict = {
        'mean_filtered_log_data': mean_filtered_log_data,
        'std_filtered_log_data': std_filtered_log_data
    }

    # --- CREATE LINSPACE BINNING ---

    linspace_binning_min_normed, linspace_binning_max_normed, linspace_binning_normed = calc_linspace_binning(
        data,
        mean_filtered_log_data,
        std_filtered_log_data,
        **settings,
    )
    linspace_binning_params = linspace_binning_min_normed, linspace_binning_max_normed, linspace_binning_normed

    # --- CALC BIN FREQUENCIES FOR OVERSAMPLING ---

    bin_frequencies = calc_bin_frequencies(
        data,
        linspace_binning_params,
        mean_filtered_log_data, std_filtered_log_data,
        **settings,
    )

    # --- INDEX CONVERSION from patch to sample ---

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

    train_sample_coords = patch_indices_to_sample_coords(
        data_shortened,
        train_valid_datetime_idx_permuts,
        y_target, x_target,
        y_input, x_input,
        y_input_padding, x_input_padding,
    )

    val_sample_coords = patch_indices_to_sample_coords(
        data_shortened,
        val_valid_datetime_idx_permuts,
        y_target, x_target,
        y_input, x_input,
        y_input_padding, x_input_padding,
    )

    # --- CREATE OVERSAMPLING ---
    # THIS USES NUMPY! NOT OPTIMIZED FOR CHUNKING!
    train_oversampling_weights, val_oversampling_weights = create_oversampling_weights(
        (train_valid_datetime_idx_permuts, val_valid_datetime_idx_permuts),
        patches,
        bin_frequencies,
        linspace_binning_params,
        mean_filtered_log_data,
        std_filtered_log_data,
        **settings
    )

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
        s_train_steps_per_epoch,
        s_val_steps_per_epoch,
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

    if s_train_steps_per_epoch is not None:
        train_steps_per_epoch = s_train_steps_per_epoch
    else:
        train_steps_per_epoch = len(train_data_set)

    if s_val_steps_per_epoch is not None:
        val_steps_per_epoch = s_val_steps_per_epoch
    else:
        val_steps_per_epoch = len(val_data_set)

    # TODO: Use Log weights instead (--> apple note 'Bin Frequencies for Oversampling in xarray')

    train_weighted_random_sampler = WeightedRandomSampler(weights=train_oversampling_weights, # TODO: LOG WEIGHTS BETTER?
                                                          num_samples=train_steps_per_epoch,
                                                          replacement=True)

    val_weighted_random_sampler = WeightedRandomSampler(weights=val_oversampling_weights,
                                                        num_samples=val_steps_per_epoch,
                                                        replacement=True)

    # This assumes same order in weights as in data set
    # replacement=True allows for oversampling and in exchange not showing all samples each epoch
    # num_samples gives number of samples per epoch.
    # Setting to len data_set forces sampler to not show all samples each epoch

    # Training data loader
    train_data_loader = DataLoader(
        train_data_set,
        sampler=train_weighted_random_sampler,
        # shuffle=True,  # remove this when using the random sampler
        batch_size=s_batch_size,
        drop_last=True,
        num_workers=s_num_workers_data_loader,
        pin_memory=True
    )

    # Validation data loader
    if s_oversample_validation:
        # ... with oversampling:
        validation_data_loader = DataLoader(
            train_data_set,
            sampler=val_weighted_random_sampler,
            batch_size=s_batch_size,
            drop_last=True,
            num_workers=s_num_workers_data_loader,
            pin_memory=True)
    else:
        # ... without oversampling:
        validation_data_loader = DataLoader(
            val_data_set,
            batch_size=s_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=s_num_workers_data_loader,
            pin_memory=True)

    print('Num training batches: {} \nNum validation Batches: {} \nBatch size: {}'.format(len(train_data_loader),
                                                                                       len(validation_data_loader),
                                                                                       s_batch_size))

    return train_data_loader, validation_data_loader, train_steps_per_epoch, val_steps_per_epoch


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
        s_calc_baseline, **__
):
    """
    All the junk surrounding train_l() goes in here
    Please keep intput arguments in the same order as the output of create_data_loaders()
    """

    train_logger, val_logger, base_train_logger, base_val_logger = create_loggers(**settings)

    # This is used to save checkpoints of the model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=s_dirs['model_dir'],
        monitor='val_mean_loss',
        filename='model_{epoch:04d}_{val_mean_loss:.2f}_best',
        save_top_k=1,
        save_last=True
    )
    # save_top_k=-1, prevents callback from overwriting previous checkpoints

    if s_profiling:
        profiler = PyTorchProfiler(dirpath=s_dirs['profile_dir'], export_to_chrome=True)
    else:
        profiler = None

    # Increase time out for weights and biases to prevent time out on galavani
    os.environ["WANDB__SERVICE_WAIT"] = "600"

    # Different project names depending on mode:
    if s_local_machine_mode:
        wandb_project_name = 'local_testing'
    else:
        wandb_project_name = 'cluster_runs'

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
        callbacks=callback_list,
        profiler=profiler,
        max_epochs=s_max_epochs,
        log_every_n_steps=1,
        logger=logger,
        devices=s_num_gpus,
        check_val_every_n_epoch=s_check_val_every_n_epoch,
        strategy='ddp',
        num_sanity_val_steps=0
    )
    # num_sanity_val_steps=0 turns off validation sanity checking
     # precision='16-mixed'
    # 'devices' argument is ignored when device == 'cpu'
    # Speed up advice: https://pytorch-lightning.readthedocs.io/en/1.8.6/guides/speed.html

    # trainer.logger = logger
    trainer.fit(model_l, train_data_loader, validation_data_loader)

    # Network_l instance is returned to be able to plot lr_scheduler
    return model_l


def create_s_dirs(sim_name, s_local_machine_mode):

    s_dirs = {}
    if s_local_machine_mode:
        s_dirs['save_dir'] = 'runs/{}'.format(sim_name)
        s_dirs['prediction_dir'] = f'/home/jan/Programming/weather_data/predictions/{sim_name}'

    else:
        s_dirs['save_dir'] = '/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs/{}'.format(sim_name)
        s_dirs['prediction_dir'] = f'/mnt/qb/work2/butz1/bst981/weather_data/predictions/{sim_name}'


    # s_dirs['save_dir'] = 'runs/{}'.format(s_sim_name)
    s_dirs['plot_dir'] = '{}/plots'.format(s_dirs['save_dir'])
    s_dirs['plot_dir_images'] = '{}/images'.format(s_dirs['plot_dir'])
    s_dirs['plot_dir_fss'] = '{}/fss'.format(s_dirs['plot_dir'])
    s_dirs['model_dir'] = '{}/model'.format(s_dirs['save_dir'])
    s_dirs['code_dir'] = '{}/code'.format(s_dirs['save_dir'])
    s_dirs['profile_dir'] = '{}/profile'.format(s_dirs['save_dir'])
    s_dirs['logs'] = '{}/logs'.format(s_dirs['save_dir'])
    s_dirs['data_dir'] = '{}/data'.format(s_dirs['save_dir'])

    return s_dirs


if __name__ == '__main__':

    s_local_machine_mode = True

    s_force_data_preprocessing = False  # This forces data preprocessing instead of attempting to load preprocessed data

    s_sim_name_suffix = 'debug_10_epochs'  # 'bernstein_scheduler_0_1_0_5_1_2' #'no_gaussian_blurring__run_3_with_lt_schedule_100_epoch_eval_inv_normalized_eval' # 'No_Gaussian_blurring_with_lr_schedule_64_bins' #'sigma_init_5_exp_sigma_schedule_WITH_lr_schedule_xentropy_loss_20_min_lead_time'#'scheduled_sigma_exp_init_50_no_lr_schedule_100G_mem' #'sigma_50_no_sigma_schedule_no_lr_schedule' #'scheduled_sigma_exp_init_50_no_lr_schedule_100G_mem'# 'sigma_50_no_sigma_schedule_lr_init_0_001' # 'scheduled_sigma_exp_init_50_lr_init_0_001' #'no_gaussian_smoothing_lr_init_0_001' #'' #'scheduled_sigma_exp_init_50_lr_init_0_001' #'no_gaussian_smoothing_lr_init_0_001' #'scheduled_sigma_cos_init_20_to_0_1_lr_init_0_001' #'smoothing_constant_sigma_1_and_lr_schedule' #'scheduled_sigma_cos_init_20_to_0_1_lr_init_0_001'

    # Getting rid of all special characters except underscores
    s_sim_name_suffix = no_special_characters(s_sim_name_suffix)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if s_local_machine_mode:
        s_sim_name = 'Run_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        s_sim_name = 'Run_{}_ID_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                           int(os.environ['SLURM_JOB_ID']))  # SLURM_ARRAY_TASK_ID

    s_sim_name = s_sim_name + s_sim_name_suffix

    s_dirs = create_s_dirs(s_sim_name, s_local_machine_mode)

    settings = \
        {
            's_local_machine_mode': s_local_machine_mode,
            's_force_data_preprocessing': s_force_data_preprocessing,
            's_sim_name': s_sim_name,
            's_sim_same_suffix': s_sim_name_suffix,

            's_convnext': True,  # Use ConvNeXt instead of ours

            's_plotting_only': False,  # If active loads sim s_plot_sim_name and runs plotting pipeline
            's_plot_sim_name': 'Run_20241105-182147_ID_774405training_and_zarr_saving_50_epochs', # 'Run_20240620-174257_ID_430381default_switching_region_32_bins_100mm_conv_next_fixed_logging_and_linspace_binning',  # _2_4_8_16_with_plotting_fixed_plotting', #'Run_20231005-144022TEST_several_sigmas_2_4_8_16_with_plotting_fixed_plotting',

            # Save data loader variables
            's_save_prefix_data_loader_vars': 'data_loader_vars_dec_24',
            's_data_loader_vars_path': '/mnt/qb/work2/butz1/bst981/weather_data/data_loader_vars',

            # Max number of frames in proccessed data set for debugging (validation + training)
            's_max_num_filter_hits': None,  # [Disabled when set to None]

            's_max_epochs': 50, #100,  #10  # default: 50 Max number of epochs, affects scheduler (if None: runs infinitely, does not work with scheduler)
            #  In case only a specific time period of data should be used i.e.: ['2021-01-01T00:00', '2021-01-01T05:00']
            #  Otherwise set to None
            's_crop_data_time_span': ['2019-01-01T00:00', '2019-03-01T00:00'], #['2019-01-01T00:00', '2019-02-01T00:00'],  # Influences RAM usage. This can also be 'None'

            # Splitting training / validation
            's_split_chunk_duration': '1D',
            # The time duration of the chunks (1D --> 1 day, 1h --> 1 hour), goes into dataset.resample
            's_ratio_train_val_test': (0.7, 0.15, 0.15),
            # These are the splitting ratios between (train, val, test), adding up to 1
            's_split_seed': 42,
            # This is the seed that the train / validation split is generated from (only applies to training of exactly the same time period of the data)

            # Number of steps per epoch in random sampler, can be None:
            # This basically makes the epoch notation more or less unnecessary (scheduler is also coup[led to training steps)
            # So this mainly influences how often things are logged
            's_train_steps_per_epoch': 500 * 0.7,  # Can be None
            's_val_steps_per_epoch': 500 * 0.15,  # Can be None

            # Load Radolan
            's_folder_path': '/mnt/qb/work2/butz1/bst981/weather_data/dwd_nc/zarr',  #'/mnt/qb/work2/butz1/bst981/weather_data/benchmark_data_set',
            's_data_file_name': 'RV_recalc.zarr',  #'yw_done.zarr',
            's_data_variable_name': 'RV_recalc',

            # Load DEM
            's_dem_path': '/mnt/qb/work2/butz1/bst981/weather_data/dem/dem_benchmark_dataset_1200_1100.zarr',
            's_dem_variable_name': 'dem',

            # Load baseline for evaluation:
            's_baseline_path': None,
            's_baseline_variable_name': 'extrapolation',
            's_num_input_frames_baseline': 4,

            's_num_workers_data_loader': 16,  # Should correspond to number of cpus, also increases cpu ram --> FOR DEBUGGING SET TO 0
            's_check_val_every_n_epoch': 1,  # Calculate validation every nth epoch for speed up, NOT SURE WHETHER PLOTTING CAN DEAL WITH THIS BEING LARGER THAN 1 !!

            # Parameters related to lightning
            's_num_gpus': 8,
            's_batch_size': 128, #our net on a100: 64  #48, # 2080--> 18 läuft 2080-->14 --> 7GB /10GB; v100 --> 45  55; a100 --> 64, downgraded to 45 after memory issue on v100 with smoothing stuff
            # resnet 34 original res blocks on a100 --> batch size 32 (tested 64, which did not work)
            # Make this divisible by 8 or best 8 * 2^n

            # Parameters that give the network architecture
            's_upscale_c_to': 32,  # 64, #128, # 512,
            's_num_bins_crossentropy': 32,  # 64, #256,

            # Parameters that give binning
            's_linspace_binning_cut_off_unnormalized': 100,
            # Let's cut that off ad-hoc (in mm/h) , everything above is sorted into the last bin

            # -- Everything related to the patching and space time of network input / ouyput --
            's_input_height_width': 256, # width / height of input for network
            's_input_padding': 32, # Additional padding of input for randomcrop augmentation. Dataloader returns patches of size s_input_height_width + s_input_padding
            's_target_height_width': 32, # width / height of target - this is what is used to patch the data
            's_num_input_time_steps': 4,  # The number of subsequent time steps that are used for prediction
            's_num_lead_time_steps': 3, # 0 --> 0 min prediction (target == last input) ; 1 --> 5 min predicition, 3 --> 15min etc
            # This is substracted by 2: settings['s_num_lead_time_steps'] = 's_num_lead_time_steps' -2 for following reasons:
            # 5, # The number of pictures that are skipped from last input time step to target, starts with -1
            # (starts counting at filtered_data_loader_indecies_dict['last_idx_input_sequence'], where last index is excess
            # for arange ((np.arange(1:5) = [1,2,3,4])

            # Filter conditions:
            's_filter_threshold_mm_rain_each_pixel': 0.1,  # threshold for each pixel filter condition
            's_filter_threshold_percentage_pixels': 0.5,

            's_save_trained_model': True,  # saves model every epoch
            's_load_model': False,
            's_load_model_name': 'Run_·20230220-191041',
            's_dirs': s_dirs,
            'device': device,
            's_learning_rate': 0.001,  # 0.0001
            # For some reason the lr scheduler starts one order of magnitude below the given learning rate (10^-4, when 10^-3 is given)
            's_lr_schedule': True,  # enables lr scheduler, takes s_learning_rate as initial rate

            # Loss
            's_crps_loss': False,  # CRPS loss instead of X-entropy loss

            # DLBD, Gaussian smoothing
            's_gaussian_smoothing_target': False,
            's_sigma_target_smoothing': 0.1,  # In case of scheduling this is the initial sigma
            's_schedule_sigma_smoothing': False,
            's_gaussian_smoothing_multiple_sigmas': False, # ignores s_gaussian_smoothing_target, s_sigma_target_smoothing and s_schedule_sigma_smoothing, s_schedule_multiple_sigmas activates scheduling for multiple sigmas
            's_multiple_sigmas': [0.1, 0.5, 1, 2], # FOR SCHEDULING MAKE SURE LARGEST SIGMA IS LAST, List of sigmas in case s_gaussian_smoothing_multiple_sigmas == True; to create loss mean is taken of all losses that each single sigma would reate
            # ! left most sigma prediction is the one that is plotted. Usually this is close to zero such that it is almost pixel-wise!
            's_schedule_multiple_sigmas': False, # Bernstein scheduling: Schedule multiple sigmas with bernstein polynomial,

            # Logging
            's_oversample_validation': True,  # Oversample validation just like training, such that training and validations are directly copmparable
            's_calc_baseline': False,  # Baselines are calculated and plotted --> Optical flow baseline
            's_epoch_repetitions_baseline': 1000, #TODO NO LONGER IN USE # Number of repetitions of baseline calculation; average is taken; each epoch is done on one batch by dataloader

            's_testing': True,  # Runs tests before starting training
            's_profiling': False,  # Runs profiler

            # Plotting stuff
            's_no_plotting': False,  # This sets all plotting boos below to False
            's_plot_average_preds_boo': True,
            's_plot_pixelwise_preds_boo': True,
            's_plot_target_vs_pred_boo': True,
            's_plot_mse_boo': True,
            's_plot_losses_boo': True,
            's_plot_img_histogram_boo': True,
        }

    if settings['s_local_machine_mode']:

        settings['s_plotting_only'] = True
        settings['s_plot_sim_name'] = 'Run_20241210-154712debug_10_epochs'
        settings['s_data_variable_name'] = 'RV_recalc'
        settings['s_folder_path'] = 'dwd_nc/own_test_data'
        settings['s_data_file_name'] = 'testdata_two_days_2019_01_01-02.zarr'
        settings['s_dem_path'] = '/home/jan/Programming/weather_data/dem/dem_benchmark_dataset_1200_1100.zarr'
        settings['s_baseline_path'] =   ('/home/jan/Programming/weather_data/baselines_two_days/'
                                        'testdata_two_days_2019_01_01-02_extrapolation.zarr')
        settings['s_baseline_variable_name'] = 'extrapolation'
        settings['s_num_input_frames_baseline'] = 4
        settings['s_upscale_c_to'] = 32  # 8
        settings['s_batch_size'] = 4  # 8
        settings['s_data_loader_chunk_size'] = 1
        settings['s_testing'] = True  # Runs tests at the beginning
        settings['s_num_workers_data_loader'] = 0  # Debugging only works with zero workers
        settings['s_max_epochs'] = 10  # 2
        settings['s_num_gpus'] = 1
        settings['s_crop_data_time_span'] = ['2019-01-01T08:00', '2019-01-01T09:00'] # ['2019-01-01T08:00', '2019-01-01T12:00']
        settings['s_split_chunk_duration'] = '5min' #'15min' #'1h'
        settings['s_ratio_train_val_test'] = (0.4, 0.3, 0.3)

        settings['s_train_steps_per_epoch'] = 4
        settings['s_val_steps_per_epoch'] = 4

        settings['s_multiple_sigmas'] = [2, 16]
        settings['s_data_loader_vars_path'] = '/home/jan/Programming/weather_data/data_loader_vars' #'/mnt/qb/work2/butz1/bst981/weather_data/data_loader_vars' #

        settings['s_max_num_filter_hits'] = None  # 4 # None or int

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
        load_dirs = create_s_dirs(settings['s_plot_sim_name'], settings['s_local_machine_mode'])
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
















