from load_data_xarray import (
    create_patches,
    get_index_permutations,
    patch_indecies_to_sample_coords,
    split_data_from_time_keys,
    FilteredDatasetXr
)
from helper.checkpoint_handling import load_from_checkpoint, get_checkpoint_names

from torch.utils.data import DataLoader,


def sample_coords_for_all_patches(
        train_time_keys, val_time_keys, test_time_keys,

        settings,
        s_width_height_target,
        s_width_height,
        s_split_chunk_duration,
        **__,
):
    """
    This creates returns the sample_coords for all, unfiltered patches.

    Input
        train_time_keys, val_time_keys, test_time_keys: list(np.datetime64)
            These are the time keys that determine the train, val and test sets
            They refer to the group names of patches.resample(time=s_split_chunk_duration)

    Output
        sample_coords (train, val, test): tuple(np.array): Coordinate space

            array of arrays with valid patch coordinates

            shape: tuple([num_valid_patches, num_dims=3])
            [
            [np.datetime64 target frame,
            slice of y coordinates,
            slice of x coordinates],
            ...]

            x and y coordinates refer to the coordinate system with respect to corred CRS and projection in data_shortened,
            not to lat/lon and also not to the patch coordinates _inner and _outer
    """

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
    # We do not filter, as we want to predict and resample all data

    # Resample patches by days
    patches_resampled = patches.resample(time=s_split_chunk_duration)

    # Split
    patches_train = split_data_from_time_keys(patches_resampled, train_time_keys)
    patches_val = split_data_from_time_keys(patches_resampled, val_time_keys)
    patches_test = split_data_from_time_keys(patches_resampled, test_time_keys)

    # --- From patches create sample coords ---

    # As we want all patches and do not do any filtering in this case we simply permute the _outer patch indecies

    time_dim_name, y_dim_name, x_dim_name = [
        'time',
        'y_outer',
        'x_outer',
    ]

    # Get index permutations [[time: np.datetime64, y_idx: int, x_idx: int], ...] for all patches
    index_permuts_patches_train = get_index_permutations(patches_train, time_dim_name, y_dim_name, x_dim_name)
    index_permuts_patches_val = get_index_permutations(patches_val, time_dim_name, y_dim_name, x_dim_name)
    index_permuts_patches_test = get_index_permutations(patches_test, time_dim_name, y_dim_name, x_dim_name)

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

    return train_sample_coords, val_sample_coords, test_sample_coords


def create_eval_dataloaders(
        train_sample_coords, val_sample_coords, test_sample_coords,
        radolan_statistics_dict,

        settings,
        s_batch_size,
        s_num_workers_data_loader,
        **__,
):

    train_data_set_eval = FilteredDatasetXr(
        train_sample_coords,
        radolan_statistics_dict,
        settings,
    )

    val_data_set_eval = FilteredDatasetXr(
        val_sample_coords,
        radolan_statistics_dict,
        settings,
    )

    test_data_set_eval = FilteredDatasetXr(
        test_sample_coords,
        radolan_statistics_dict,
        settings,
    )

    train_data_loader_eval = DataLoader(
        train_data_set_eval,
        shuffle=False,
        batch_size=s_batch_size,
        drop_last=True,
        num_workers=s_num_workers_data_loader,
        pin_memory=True
    )

    val_data_loader_eval = DataLoader(
        val_data_set_eval,
        shuffle=False,
        batch_size=s_batch_size,
        drop_last=True,
        num_workers=s_num_workers_data_loader,
        pin_memory=True
    )

    test_data_loader_eval = DataLoader(
        test_data_set_eval,
        shuffle=False,
        batch_size=s_batch_size,
        drop_last=True,
        num_workers=s_num_workers_data_loader,
        pin_memory=True
    )

    return train_data_loader_eval, val_data_loader_eval, test_data_loader_eval

def ckpt_to_pred(
        train_time_keys, val_time_keys, test_time_keys,
        radolan_statistics_dict,

        ckp_settings,  # Make sure to pass the settings of the checkpoint
        s_dirs,
        **__,
):
    """
    This creates a .zarr file for all predictions of the model checkpoints

    Input
        train_time_keys, val_time_keys, test_time_keys: list(np.datetime64)
            These are the time keys that determine the train, val and test sets
            They refer to the group names of patches.resample(time=s_split_chunk_duration)

        radolan_statistics_dict: dict
            Radolan satatistics for normalization, that has been calculated on the filtered target patches
            For all other weather variables are the normalization statistics are calculated on-the-fly on the
            whole dataset

        ckp_settings: dict
                ckp_settings are the settings of the run that the checkpoint was created with.
                The entries of settings are expected to start with s_...
                Make sure to modify settings that influence the forward pass according to your wishes
                This is particularly true for the entries:
                s_device
                s_num_gpus
    """
    save_dir = s_dirs['save_dir']

    checkpoint_names = get_checkpoint_names(save_dir)
    for checkpoint_name in checkpoint_names:

        train_sample_coords, val_sample_coords, test_sample_coords = sample_coords_for_all_patches(
            train_time_keys, val_time_keys, test_time_keys,
            **ckp_settings,
        )

        train_data_loader_eval, val_data_loader_eval, test_data_loader_eval = create_eval_dataloaders(
            train_sample_coords, val_sample_coords, test_sample_coords,
            radolan_statistics_dict,
            **ckp_settings,
        )

        load_from_checkpoint(
            save_dir,
            checkpoint_name,

            **ckp_settings,
        )

















