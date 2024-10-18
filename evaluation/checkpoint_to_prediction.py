from load_data_xarray import (
    create_patches,
    get_index_permutations,
    patch_indecies_to_sample_coords,
    split_data_from_time_keys,
)

def ckp_to_pred(
        train_time_keys, val_time_keys, test_time_keys,

        settings,
        s_width_height_target,
        s_width_height,
        s_split_chunk_duration,
        **__,
):

    # Define constants for pre-processing
    y_target, x_target = s_width_height_target, s_width_height_target  # 73, 137 # how many pixels in y and x direction
    y_input, x_input = s_width_height, s_width_height
    y_input_padding, x_input_padding = 0, 0  # No augmentation, thus no padding for evaluation

    # --- Load patches ---

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
        y_target,
        x_target,
        **settings
    )

    # --- Split patches ---
    # We do not filter, as we want to predict and reassemble all data

    # Resample patches by days
    patches_resampled = patches.resample(time=s_split_chunk_duration)

    # Split
    patches_train = split_data_from_time_keys(patches_resampled, train_time_keys)
    patches_val = split_data_from_time_keys(patches_resampled, val_time_keys)
    patches_test = split_data_from_time_keys(patches_resampled, test_time_keys)

    # --- From patches create sample coords ---

    # As we want all patches and do not do any filtering in this case we simply permute the _outer patch indecies

    time_dim, y_dim, x_dim = [
        'time',
        'y_outer',
        'x_outer',
    ]

    # TODO MISTAKE: indecies of index_permuts_patches_train / val / test are used to directly load from
    # TODO data_shortened
    index_permuts_patches_train = get_index_permutations(patches_train, time_dim, y_dim, x_dim)
    index_permuts_patches_val = get_index_permutations(patches_val, time_dim, y_dim, x_dim)
    index_permuts_patches_test = get_index_permutations(patches_test, time_dim, y_dim, x_dim)


    # --- Check for duplicates ---
    # Combine all sample coordinates
    all_sample_coords = index_permuts_patches_train + index_permuts_patches_val + index_permuts_patches_test

    # Calculate the total number of samples and the number of unique samples
    total_samples = len(all_sample_coords)
    unique_samples = len(set(all_sample_coords))

    # Check for duplicates
    if total_samples != unique_samples:
        num_duplicates = total_samples - unique_samples
        raise ValueError(
            f'There are {num_duplicates} duplicates among train, val, and test sets.'
        )

    # ... and calculate the sample coords with respect to the CRS and projection of data_shortened of them
    train_sample_coords = patch_indecies_to_sample_coords(
        data_shortened,
        index_permuts_patches_train,
        y_target, x_target,
        y_input, x_input,
        y_input_padding, x_input_padding,
    )

    val_sample_coords = patch_indecies_to_sample_coords(
        data_shortened,
        index_permuts_patches_val,
        y_target, x_target,
        y_input, x_input,
        y_input_padding, x_input_padding,
    )

    test_sample_coords = patch_indecies_to_sample_coords(
        data_shortened,
        index_permuts_patches_test,
        y_target, x_target,
        y_input, x_input,
        y_input_padding, x_input_padding,
    )
    








