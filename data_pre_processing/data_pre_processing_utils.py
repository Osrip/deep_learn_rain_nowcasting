import numpy as np
import xarray as xr
import itertools

from xarray.core.groupby import DatasetGroupBy
import random
from helper.pre_process_target_input import normalize_data, inverse_normalize_data
import torch


def create_patches(
        y_target, x_target,

        s_num_input_time_steps,
        s_num_lead_time_steps,
        s_folder_path,
        s_data_file_name,
        s_data_variable_name,
        s_crop_data_time_span,
        **__,
):
    """
    Patch data into patches of the target size
    s_crop_data_time_span determines whether the original data is cropped that the patching is created from
    Input:
        y_target, x_target: int, int
            The size of the target patch in pixels (y_target, x_target = s_target_height_width, s_target_height_width)
        s_num_input_time_steps: int
            The number of input frames
        s_num_lead_time_steps: int
            The number of lead time steps
        s_folder_path: str
            The folder path where the data is stored
        s_data_file_name: str
            The name of the data file
        s_crop_data_time_span: tuple(np.datetime64, np.datetime64)
            The time span that is used for the data, if None the whole data
    Output:
        patches: xr.Dataset
            xr.Dataset Patch dimensions y_outer, x_outer give one coordinate pair for each patch,
            y_inner, x_inner give pixel dimensions for each patch
        data: xr.Dataset
            The unpatched data that has global pixel coordinates
        data_shortened: xr.Dataset
            same as data, but beginning is missing (lead_time + num input frames) such that we can go
            'back in time' to go from target time to input time.
    """

    # Loading data into xarray
    load_path = '{}/{}'.format(s_folder_path, s_data_file_name)

    data = xr.open_zarr(load_path, chunks={'step': 1, 'time': 1, 'y': 1200, 'x': 1100},
                           decode_timedelta=True)

    if s_crop_data_time_span is not None:
        start_time, stop_time = np.datetime64(s_crop_data_time_span[0]), np.datetime64(s_crop_data_time_span[1])
        crop_slice = slice(start_time, stop_time)
        data = data.sel(time=crop_slice)

    data = data.squeeze()

    # Set negative values to 0
    # I realized that there are quite a few extremely tiny negative values in the data (WHY?), set those to zero
    data_min = data.min(skipna=True, dim=None)[s_data_variable_name].values
    if data_min < -0.1:
        raise ValueError(f'The min value of the data is {data_min}, which is below the threshold of -0.1')
    data = data.where((data >= 0) | np.isnan(data), other=0)  #All values that are NOT chosen ( >= 0 or nan) are set to 0

    # Cut off the beginning  of the data as the time length, that one sample has (input frames + lead time + target)
    # as we will 'go back in time' to generate the inputs from the target patches
    data_shortened = data.isel(
        time=slice(s_num_input_time_steps + s_num_lead_time_steps, -1)
    )

    # partition the data into pt x y_target x x_target blocks using coarsen --> construct DatasetCoarsen object
    # In this implementation each target corresponds to one patch
    coarse = data_shortened.coarsen(
        y=y_target,
        x=x_target,
        # time = 1, # No patching along time dimension
        side="left",  # "left" means that the blocks are aligned to the left of the input
        boundary="trim"  # boundary="trim" removes the last block if it is too small
    )

    # construct a new data set, where the patches are folded into a new dimension
    patches = coarse.construct(
        y=("y_outer", "y_inner"),  # Those are the patche indecies / the patch dimesnion,
                                   # each index pair corresponds to one patch
        x=("x_outer", "x_inner"),  # Those are the pixel dimensions of the patches
        # keep_attrs=True          # This would keep attrs (not coordinates!) not needed here.
    )
    # The coordinates are killed this way for the _inner and _outer dims
    # Dimensions without coordinates: y_outer, y_inner, x_outer, x_inner

    return patches, data, data_shortened

def filter_patches(
        patches,

        s_filter_threshold_mm_rain_each_pixel,
        s_filter_threshold_percentage_pixels,
        **__,
):
    """
    Called after create_patches()
    Filter patches. Patches that passed filter are called 'valid'.
    Input:
        patches: xr.Dataset
            The patches that were created by create_patches()
            xr.Dataset Patch dimensions y_outer, x_outer give one coordinate pair for each patch,
            y_inner, x_inner give pixel dimensions for each patch
        s_filter_threshold_mm_rain_each_pixel: float
            The threshold for the rain in mm that each pixel has to exceed
        s_filter_threshold_percentage_pixels: float
            The percentage of pixels that have to exceed the threshold
    Output:
        valid_patches_boo: xr.Dataset
            Boolean xr.Dataset with y_outer and x_outer that defines the valid patches
    """

    # Replace NaNs with 0s for the filter (Alternatively we could also throw out all targets with NaNs in them)
    patches_no_nan = patches.fillna(0)

    # --- FILTER ---
    # define a threshold for each pixel --> we get a pixel-wise boo
    patches_boolean_pixelwise = patches_no_nan > s_filter_threshold_mm_rain_each_pixel
    # We are calculating the percentage of pixels that passed filter (mean of boolean gives percentage of True)
    # --> we are getting rid of the patch dimensions y_inner and x_inner,
    patches_percentage_pixels_passed = patches_boolean_pixelwise.mean(("y_inner", "x_inner"))  # , "time_inner"
    # Now we are creating a boolean again by comparing the percentage of pixels that exceed the rain threshold to the minimally required
    # percentage of pixels that exceed the rain threshold
    # --> valid_patches includes only the y_outer and x_outer, where each pair of indecies represents one patch. Its values are boolean, indicating whether or not the outer indecies correspond to a valid or invalid patch.
    valid_patches_boo = patches_percentage_pixels_passed > s_filter_threshold_percentage_pixels

    return valid_patches_boo


def patches_boo_to_datetime_idx_permuts(
        valid_patches_boo,

        s_data_variable_name,
        **__,
):
    '''
    Extracts the time coordinate (np.datetime64) and spatial indices (y_outer, x_outer) for valid patches from xr.dataset.

    Input
        valid_patches_boo: xr.Dataset
            Boolean dataset indicating valid patches, with dimensions `time`, `y_outer`, and `x_outer`.
        s_data_variable_name: str
            Name of the data variable containing the boolean mask.

    Output
        valid_datetime_idx_permuts: list of tuples
            A list of tuples (time, y_outer, x_outer) for each valid patch:
                - `np.datetime64 time`: Time coordinate of the valid patch.
                - `int y_outer`: Spatial index in the y-direction.
                - `int x_outer`: Spatial index in the x-direction.

    Example
        Returns: [(time, y_idx, x_idx), ...] for each valid patch.
    '''

    valid_patches_da = valid_patches_boo[s_data_variable_name]
    indices = np.nonzero(valid_patches_da.values)
    times = valid_patches_da.coords['time'].values[indices[0]]
    y_indices = indices[1]
    x_indices = indices[2]
    valid_datetime_idx_permuts = list(zip(times, y_indices, x_indices))
    return valid_datetime_idx_permuts


def patch_indices_to_sample_coords(
        data_shortened,
        valid_samples,
        y_target, x_target,
        y_input, x_input,
        y_input_padding, x_input_padding
) -> np.array(tuple):
    '''
    This functions converts the 'valid_samples' which give the outer indecies with respect to 'patches',
    directly recalculates those to indices with respect to data_shortened and returns global coordinates that refer
    to any data that has the same CRS and projection as data_shortened

    All patches, where the input frame exceeds the size of data_shortened is dropped at the moment! Keep in mind
    that data_shortened has a quite large nan padding by default at the moment

    Input
        data_shortened: xr.dataset
            The raw precipitation data, where the first entries have been cut along time dim (according to the length of lead time)
            As we are always using the target frame as a time reference. Thus data shortened starts at the time
            of the first target frame.
        valid_samples: list, outer index space (patch indecies), time in datetime
            list that includes all index permutation of the valid patches (those patches that passed the filter)

            shape: [num_valid_patches, num_dims=3]
            [
            [time datetime, ! This is in coordinate datetime formet, not in index space !
            y_outer idx,    with respect to patches (directly recalculated to idx in data shortened)
            x_outer idx],    with respect to patches (directly recalculated to idx in data shortened)
            ...]

            All those are _outer indecies (so ints) and not coordinates
            If all patches should be converted use get_index_permutations() to create all possible index permutations
            of patches dataset along dims = time, y_outer, x_outer
        y_target, x_target: float(), pixel index space
            target size
        y_input, x_input: float(), pixel index space
            network input size
        y_input_padding, x_input_padding: float(), pixel index space
            input padding

    Output
        sample_coords: np.array: Coordinate space
            array of arrays with valid patch coordinates

            shape: [num_valid_patches, num_dims=3]
            [
            [np.datetime64 target frame,
            slice of y coordinates,
            slice of x coordinates],
            ...]

            x and y coordinates refer to the coordinate system with respect to correct CRS and projection in data_shortened,
            not to lat/lon and also not to the patch coordinates _inner and _outer

        valid_samples_updated:
            samples that exceeded the spatial bounds of data_shortened were dropped
            Otherwise data structure is just as in input
    '''

    y_input_padded = y_input + y_input_padding
    x_input_padded = x_input + x_input_padding

    valid_target_indecies_global = []
    valid_center_indecies_global = []
    valid_input_indecies_global = []
    sample_coords = []  # These are defined

    valid_samples_updated = []

    num_inputs_exceeding_bounds = 0

    for i, (time_datetime, y_outer_idx, x_outer_idx) in enumerate(valid_samples):

        y_global_upper = y_outer_idx * y_target
        x_global_left = x_outer_idx * x_target

        # Calculate the global indecies / slices for the targets
        slice_y_global = slice(y_global_upper, y_global_upper + y_target)
        slice_x_global = slice(x_global_left, x_global_left + x_target)
        target_slices = [time_datetime, slice_y_global, slice_x_global]

        # Calculate indices of the patch's center pixels
        center_y_global = y_global_upper + y_target // 2
        center_x_global = x_global_left + x_target // 2
        global_center_indecies = [time_datetime, center_y_global, center_x_global]

        # Calculate the global slices for input
        y_slice_input = slice(center_y_global - (y_input_padded // 2), center_y_global + (y_input_padded // 2))
        x_slice_input = slice(center_x_global - (x_input_padded // 2), center_x_global + (x_input_padded // 2))
        input_slices = [time_datetime, y_slice_input, x_slice_input]

        # Check if the larger input exceeds size, if not append the patch indecies / slices to the list
        if (
                y_slice_input.start < 0
                or y_slice_input.stop >= data_shortened.sizes['y']
                or x_slice_input.start < 0
                or x_slice_input.stop >= data_shortened.sizes['x']
        ):
            num_inputs_exceeding_bounds += 1
            continue  # Skips the rest of the code in the current loop iteration if input frame exceeds dataset bounds

        # --- Convert spatial indices to coordinates ---
        # No need to convert time as this is already in correct format.

        y_coords_input = data_shortened.y.isel(y=y_slice_input).values
        x_coords_input = data_shortened.x.isel(x=x_slice_input).values

        y_coords_slices_input = slice(y_coords_input[0], y_coords_input[-1])
        x_coords_slices_input = slice(x_coords_input[0], x_coords_input[-1])

        coords_one_sample = [time_datetime, y_coords_slices_input, x_coords_slices_input]

        valid_target_indecies_global.append(target_slices)
        valid_center_indecies_global.append(global_center_indecies)
        valid_input_indecies_global.append(input_slices)
        sample_coords.append(coords_one_sample)
        valid_samples_updated.append((time_datetime, y_outer_idx, x_outer_idx))

    print(f'{num_inputs_exceeding_bounds} patches dropped as padded input exceeded spatial data bounds')

    if not len(sample_coords) == len(valid_samples_updated):
        raise ValueError('valid_samples_updated has a different length than sample_coords')

    return np.array(sample_coords), valid_samples_updated


def all_patches_to_datetime_idx_permuts(
        patches_dataset: xr.Dataset,
        time_dim_name: str,
        y_outer_name: str,
        x_outer_name: str,
) -> list[tuple[np.datetime64, int, int]]:
    """
    Generate all index permutations for the specified dimensions of an xarray dataset, where the time dimension
    uses actual time coordinates (`np.datetime64`), and the spatial dimensions use indices.

    Input:
        dataset: xr.Dataset
            The xarray dataset containing the dimensions and coordinates.
        time_dim: str
            The name of the time dimension in the dataset.
        y_outer_name: str
            The name of the y (vertical) spatial dimension indicating the patch index.
        x_outer_name: str
            The name of the x (horizontal) spatial dimension indicating the patch index.

    Output:
        index_permutations: List[Tuple[np.datetime64, int, int]]
            A list containing all permutations of time coordinates and spatial indices.
            Each element is a tuple in the format:
                (time coordinate as np.datetime64, y index as int, x index as int)
    """
    # Retrieve the time coordinates (as np.datetime64)
    times = patches_dataset.coords[time_dim_name].values  # This should be an array of np.datetime64

    # Generate index arrays for the spatial dimensions
    y_indices = np.arange(patches_dataset.sizes[y_outer_name])
    x_indices = np.arange(patches_dataset.sizes[x_outer_name])

    # Generate all permutations using itertools.product
    index_permutations = itertools.product(times, y_indices, x_indices)

    # Convert the permutations into a list of tuples
    return list(index_permutations)


def create_split_time_keys(
        unfiltered_data_resampled: DatasetGroupBy,

        s_split_seed: int,
        s_ratio_train_val_test: tuple,
        **__,
):
    """
    This function splits the dataset. As an input it expects the unfiltered_data that is already grouped by
    the time chunks we are using (i.e. 1D --> 1 day). It then returns the (time) keys for the groups that are in the
    train / test / val set.
    These keys can be applied to create the according train / val / test set from the unfiltered but also filtered data.
    When handling filtered data, please watch out that some time keys might be not existent in the filtered data,
    as all data that would fall under this time key has been filtered out.

    Input:
        unfiltered_data: DatasetGroupby
            This is the unfiltered data, which is grouped by the time chunks we are splitting on (i.e. 1 day)
            We use the unfiltered data, as we may want to do predictions with the unfiltered data,
            then we want our validation and training sets to cover all of thgese time stamps
            ! In practice make sure to use the unshortened data, where the beginning that we cannot predict but need to !
            ! infer the predictions is not cut off !

        s_ratio_train_val_test: tuple of floats (train_ration, val_ratio, test_ratio)
            These are the splitting ratios between (train, val, test), adding up to 1
    Output:
        train_time_keys, val_time_keys, test_time_keys: List of np.datetime64
            These are the keys to select the entries of the resampled DatasetGroupBy onjects from

    Example of creating input data:
        resampled_data = data.resample(time=s_split_chunk_duration)

    Example of creating data from the keys:
        training_data = xr.concat([resampled_data[time_key] for time_key in train_time_keys])
    """
    # Ensure ratios add up to 1
    total_ratio = sum(s_ratio_train_val_test)
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError("Ratios must sum to 1.")

    # Set random seed.
    rng = random.Random(s_split_seed)

    # Shuffle the time keys
    time_keys = list(unfiltered_data_resampled.groups.keys())
    rng.shuffle(time_keys)

    N = len(time_keys)

    # Compute the counts for each split
    ratios = s_ratio_train_val_test
    n_train = int(N * ratios[0])
    n_val = int(N * ratios[1])
    n_test = N - n_train - n_val  # Ensure total sums to N

    # Adjust counts if necessary
    counts = [n_train, n_val, n_test]

    # Compute the split indices
    indices = [0]
    for count in counts:
        indices.append(indices[-1] + count)

    # Split the time keys
    train_time_keys = time_keys[:indices[1]]
    val_time_keys = time_keys[indices[1]:indices[2]]
    test_time_keys = time_keys[indices[2]:]

    # Check that each split has entries
    if not train_time_keys:
        raise ValueError("SPLIT RATIOS TOO SMALL: Train time keys are empty. Check your data or split ratios.")
    if not val_time_keys:
        raise ValueError("SPLIT RATIOS TOO SMALL: Validation time keys are empty. Check your data or split ratios.")
    if not test_time_keys:
        raise ValueError("SPLIT RATIOS TOO SMALL: Test time keys are empty. Check your data or split ratios.")

    return train_time_keys, val_time_keys, test_time_keys


def split_data_from_time_keys(
        resampled_data: DatasetGroupBy,
        time_keys,
):
    """
    This comes into play when time_keys has been created by create_split_time_stamps with data that is different from
    the data that you want to split.

    Does the same as xr.concat([resampled_valid_patches_boo[time_key] for time_key in train_time_keys], dim='time')
    just checks whether the keys are existent in resampled_data
    """
    keys_not_found = []

    group_list = []
    for time_key in time_keys:
        try:
            group_list.append(resampled_data[time_key])
        except KeyError:
            keys_not_found.append(time_key)

    print(f'split_data_from_time_keys: {len(keys_not_found)} time_keys were not present in the resampled data set'
          f'This is expected, as the data that the time_keys have been created from may be different from'
          f'the data that you are trying to split and can be ignored.')

    return xr.concat(group_list, dim='time')


def calc_statistics_on_valid_patches(
        patches: xr.Dataset,
        valid_patches_boo: xr.Dataset,

        s_data_variable_name,
        **__,
):
    '''
    Calculates 1st and 2nd moment of all valid batches
    This ignores NaNs, flattens the complete data and assumes log1p = log(x+1) for log moments
    '''
    # Compute the boolean mask if it's a Dask array
    valid_patches_boo = valid_patches_boo.compute()
    
    # Only select the filtered patches by using the boolean mask (on data that includes NaNs)
    only_filtered_data = patches.where(valid_patches_boo,
                                       drop=True)  # I checked that this actually reduces length of time, y_outer and x_outer

    # Calculate the mean and standard deviation across all dimensions, ignoring NaNs
    mean = only_filtered_data.mean(dim=None, skipna=True)
    std = only_filtered_data.std(dim=None, skipna=True)

    only_filtered_data_log = np.log1p(only_filtered_data)

    log_mean = only_filtered_data_log.mean(dim=None, skipna=True)
    log_std = only_filtered_data_log.std(dim=None, skipna=True)

    return (
        float(mean[s_data_variable_name].values),
        float(std[s_data_variable_name].values),
        float(log_mean[s_data_variable_name].values),
        float(log_std[s_data_variable_name].values)
    )


def calc_linspace_binning(
        data: xr.Dataset,
        mean_filtered_log_data,
        std_filtered_log_data,

        s_linspace_binning_cut_off_unnormalized,
        s_num_bins_crossentropy,
        s_data_variable_name,
        **__,
):
    '''
    Creates a linspace binning in normalized space of the data. The bin values are also normalized.
    Creates the vector that gives linspace binning
    Only includes the left edges of the bins.
    The
    '''
    # Calculate min and max for linspace_binning:
    binning_max_unnormed = float(data.max(dim=None, skipna=True)[s_data_variable_name].values)
    binning_min_unnormed = float(data.min(dim=None, skipna=True)[s_data_variable_name].values)

    if binning_min_unnormed != 0.0:
        raise ValueError(f'Min of precipitation data is {binning_min_unnormed} and thus below 0')

    if binning_max_unnormed < s_linspace_binning_cut_off_unnormalized:
        raise ValueError(
            f's_linspace_binning_cut_off_unnormalized is larger than the max value of the training data:,'
            f'{binning_max_unnormed} please choose a cut-off that is larger than that'
        )
        binning_max_unnormed = s_linspace_binning_cut_off_unnormalized

    # Normalize linspace binning thresholds now that data is available
    linspace_binning_min_normed = normalize_data(
        binning_min_unnormed,
        mean_filtered_log_data,
        std_filtered_log_data
    )

    linspace_binning_max_normed = normalize_data(
        binning_max_unnormed,
        mean_filtered_log_data,
        std_filtered_log_data
    )

    linspace_binning_cut_off_normed = normalize_data(
        s_linspace_binning_cut_off_unnormalized,
        mean_filtered_log_data,
        std_filtered_log_data
    )

    # Subtract a small number to account for rounding errors made in the normalization process
    linspace_binning_min_normed -= 0.001
    linspace_binning_max_normed += 0.001

    # linspace_binning only includes left bin edges. The rightmost bin egde is given by linspace binning max
    # This is used when there is cut off happening for the last bin  but the linspace binning is uniformly
    # distributed between the bounds of the data:

    linspace_binning = np.linspace(
        linspace_binning_min_normed,
        linspace_binning_cut_off_normed,
        num=s_num_bins_crossentropy,
        endpoint=True
    )  # In this case endpoint=True to get the cut-off as left bound of last bin

    # linspace_binning = np.linspace(linspace_binning_min_normed, linspace_binning_max_normed, num=s_num_bins_crossentropy,
    #                                endpoint=False)

    return linspace_binning_min_normed, linspace_binning_max_normed, linspace_binning


def calc_bin_frequencies(
        data,
        linspace_binning_params,
        mean_filtered_log_data, std_filtered_log_data,

        s_data_variable_name,
        # sample_period = '1D',
        **__,
):
    """
    This function calculates the frequencies of each bin in the data
    Extremely CPU expensive operation - therefore we are subsampling
    One hour of filtered data takes about 1 minute to calculate!

    Input:
        data: xr.DataSet
            The data over which the binning frequency shall be calculated
        # sample_period : str
        #     The period for which the bin frequencies will be calculated
        #     A pandas-compatible offset string (e.g. '1Y', '30D') specifying how long a
        #     time slice to select. If None, use entire dataset. If data is shorter than
        #     the given period, the full data is used.

    Output:
        bin_frequencies: np.array
            Gives the frequency of each bin in the same order as linspace_binning
            Sums up to 1

    To plot the bin frequencies use scripts_in_debug_mode/plot_bin_frequencies.py and call it and the end of this function
    """

    # --- Inverse normalize linspace binning as we are operating on unnormalized data in preprocessing ---

    linspace_binning_min_normed, linspace_binning_max_normed, linspace_binning_normed = linspace_binning_params

    linspace_binning_max_unnormed = inverse_normalize_data(
        linspace_binning_max_normed,
        mean_filtered_log_data,
        std_filtered_log_data
    )

    linspace_binning_unnormed = inverse_normalize_data(
        linspace_binning_normed,
        mean_filtered_log_data,
        std_filtered_log_data
    )

    # --- Determine Bin Frequencies ---
    # groupby_bins requires max to be included in the binnning
    linspace_binning_with_max_unnormed = np.append(linspace_binning_unnormed, linspace_binning_max_unnormed)

    # Subsample data
    # We are subsampling as groupby_bins is extremely CPU expensive
    # Speedup using numpy see: https://github.com/pydata/xarray/issues/6758

    # time_dim = data['time']
    # total_duration = time_dim[-1].values - time_dim[0].values
    # desired_duration = pd.to_timedelta(sample_period)
    #
    # # Only slice if desired_duration is shorter than the full dataset duration
    # if desired_duration < total_duration:
    #     data_subsampled = data.sel(time=slice(time_dim[0].values, time_dim[0].values + desired_duration))
    # else:
    #     data_subsampled = data
    # else keep data as is, since desired_duration is longer than available data

    # This eats up the vast majority of computing time (47.5s out of 47.9s for teh whole preprocessing)
    binned_data = data.groupby_bins(s_data_variable_name, linspace_binning_with_max_unnormed)
    # Count the number of values in each bin, .count() ignores NaNs.
    bin_counts = binned_data.count()[s_data_variable_name]
    # bins with 0 hits are returned as NaN.
    bin_frequencies = bin_counts / np.nansum(bin_counts)
    # convert to np.array
    bin_frequencies = bin_frequencies.values

    if not np.isclose(
            np.nansum(bin_frequencies), 1.0,
            atol=1e-2
    ):
        raise ValueError('bin_frequencies do not add up to one')

    return bin_frequencies


def create_oversampling_weights(
    list_of_valid_datetime_idx_permuts,
    patches,
    bin_frequencies,
    linspace_binning_params,
    mean_filtered_log_data,
    std_filtered_log_data,

    s_data_variable_name,
    **__
):
    """
    Creates the oversampling weights for the patches. The weights are calculated based on the bin frequencies
    and the binning of the data.

    ! THIS FUNCTION USES NUMPY ARRAYS, RAM INTENSIVE, NO EASY CHUNKING POSSIBLE !

    Input:
        list_of_valid_datetime_idx_permuts: List of valid_datetime_idx_permuts (e.g. train and val)
            valid_datetime_idx_permuts:
            A list of tuples (time, y_outer, x_outer) for each valid patch:
                - `np.datetime64 time`: Time coordinate of the valid patch.
                - `int y_outer`: Spatial index in the y-direction.
                - `int x_outer`: Spatial index in the x-direction.
        patches: xr.Dataset
            The patches dataset containing the data and the weights.
        bin_frequencies: np.array
            The frequencies of each bin in the same order as linspace_binning.
        linspace_binning_params: tuple
            Tuple containing the min, max, and normalized linspace binning.
        mean_filtered_log_data: float
            The mean of the filtered log data.
        std_filtered_log_data: float
            The standard deviation of the filtered log data.

    Output:
        list_of_oversampling_weights: List of oversampling_weights (e.g. train and val)
            oversampling_weights: List of floats
                A list of weights for each valid patch.
                Same order as valid_datetime_idx_permuts.

    To plot stuff in debug mode use scripts in: scripts_in_debug_mode/plot_bin_frequencies.py
    """
    # --- Inverse normalize linspace binning as we are operating on unnormalized data in preprocessing ---

    linspace_binning_min_normed, linspace_binning_max_normed, linspace_binning_normed = linspace_binning_params

    linspace_binning_max_unnormed = inverse_normalize_data(
        linspace_binning_max_normed,
        mean_filtered_log_data,
        std_filtered_log_data
    )

    linspace_binning_unnormed = inverse_normalize_data(
        linspace_binning_normed,
        mean_filtered_log_data,
        std_filtered_log_data
    )
    # --- Determine Bin Frequencies ---
    # TODO: Do we need TWICE THE RAM of the radolan DATASET here, as we have both the weights and the actual data?

    # ! THIS MEANS EVERYTHING IS LOADED INTO MEMORY !
    patches_np = patches[s_data_variable_name].values

    # Digitize: every pixel gets its corresponding bin index, NaNs are assigned to highest bin
    patches_digitized = np.digitize(patches_np, bins=linspace_binning_unnormed, right=False)

    # Bin weights are inverse bin frequency
    bin_weights = 1 / bin_frequencies

    # Access the bin weights from the bin indecies
    patches_weighted = bin_weights[patches_digitized-1]

    # Recreate NaNs
    patches_weighted[np.isnan(patches_np)] = np.nan

    # Assign weights to the patches ds
    patches['weights'] = xr.DataArray(
        data=patches_weighted,
        coords=patches['RV_recalc'].coords,
        dims=patches['RV_recalc'].dims,
        attrs={'description': 'Pixel-wise weights, that are used to calculate the oversampling weights'}  # Optional: add any attributes
    )

    list_of_oversampling_weights = []

    # Iterate over train and val data
    # TODO: Do the above inside the loop, less efficiency but drastically reduces memory footprint
    for valid_datetime_idx_permuts in list_of_valid_datetime_idx_permuts:
        oversampling_weights = []

        # Iterate over each sample / patch
        # Unfortunately we have to loop here, due to valid_datetime_idx_permuts, which had to be chosen as
        # .coarsen deletes all metadata of inner dimensions
        for (time_datetime, y_outer_idx, x_outer_idx) in valid_datetime_idx_permuts:
            # Doing a mix of sel and isel to handle the mix of datetime and indices
            curr_patch = patches.sel(
                time = time_datetime,
                y_outer = patches.y_outer[y_outer_idx],
                x_outer = patches.x_outer[x_outer_idx],
            )
            mean_patch_weight = np.nanmean(curr_patch['weights'])
            oversampling_weights.append(mean_patch_weight)
        list_of_oversampling_weights.append(np.array(oversampling_weights))

    return list_of_oversampling_weights


def convert_datetime64_array_to_float_tensor(datetime_array):
    """
    Converts a numpy datetime64 array into a PyTorch float tensor.

    Input
        datetime_array: np.ndarray
            An array of numpy datetime64[ns] objects.

    Output
        float_tensor: torch.Tensor
            A 1D PyTorch tensor where each datetime64 is converted to a float timestamp.
    """
    # Convert each datetime64 to float timestamp
    float_array = datetime_array.astype('float64')  # datetime64[ns] to float64
    # Convert the numpy array to a PyTorch tensor
    float_tensor = torch.tensor(float_array, dtype=torch.float64)
    return float_tensor


def convert_float_tensor_to_datetime64_array(float_tensor):
    """
    Converts a PyTorch float tensor back into a numpy datetime64 array.

    Input
        float_tensor (torch.Tensor): A 1D PyTorch tensor where each value represents a timestamp as float.

    Output
        np.ndarray: A numpy array of datetime64[ns] objects reconstructed from the float tensor.
    """
    # Convert tensor back to numpy array of floats
    float_array = float_tensor.numpy()
    # Convert float timestamps back to datetime64[ns]
    datetime_array = float_array.astype('datetime64[ns]')
    return datetime_array