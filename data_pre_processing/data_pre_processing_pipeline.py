import time

import numpy as np
import xarray as xr

from helper.memory_logging import format_ram_usage, format_duration
from data_pre_processing.data_pre_processing_utils import create_patches, filter_patches, create_split_time_keys, split_data_from_time_keys, \
    calc_statistics_on_valid_patches, calc_linspace_binning, calc_bin_frequencies, patches_boo_to_datetime_idx_permuts, \
    patch_indices_to_sample_coords, create_oversampling_weights


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
        s_oversampling_enabled=True,  # Default to True for backward compatibility
        **__,
):
    '''
    Patches refer to targets
    Samples refer to the data delivered by data loader
    '''
    print('Preprocess Radolan data:')
    start_time = time.time()
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
    if s_oversampling_enabled:
        print(f"Calculate bin frequencies for oversampling ... {format_ram_usage()}"); step_start_time = time.time()

        # Load specific time span to calculate bin frequencies on - quick & dirty
        # as this calculation is extremely expensive
        load_path = '{}/{}'.format(s_folder_path, s_data_file_name)
        data_set = xr.open_zarr(load_path, chunks=None)
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
    else:
        print(f"Skipping bin frequencies calculation (oversampling disabled)")
        bin_frequencies = None

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
    if s_oversampling_enabled:
        print(f"Create oversampling weights ... {format_ram_usage()}");
        step_start_time = time.time()
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
    else:
        print(f"Skipping oversampling weights calculation (oversampling disabled)")
        train_oversampling_weights, val_oversampling_weights = None, None

    print(f"Preprocessing complete. Total time: {format_duration(time.time() - start_time)}")

    return (
        train_sample_coords, val_sample_coords,
        train_time_keys, val_time_keys, test_time_keys,
        train_oversampling_weights, val_oversampling_weights,
        radolan_statistics_dict,
        linspace_binning_params,
    )
