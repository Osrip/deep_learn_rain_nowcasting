import numpy as np
import xarray as xr
from xarray.core.groupby import DatasetGroupBy
import random
from torch.utils.data import Dataset
from helper.pre_process_target_input import normalize_data
import torch
from torchvision import transforms


class FilteredDatasetXr(Dataset):
    def __init__(self, sample_coords, radolan_statistics_dict, settings):
        # super().__init__()
        self.sample_coords = sample_coords
        self.settings = settings

        s_folder_path = settings['s_folder_path']
        s_dem_path = settings['s_dem_path']
        s_data_file_name = settings['s_data_file_name']

        s_data_variable_name = settings['s_data_variable_name']
        s_dem_variable_name = settings['s_dem_variable_name']


        # --- load data ---
        # Radolan
        load_path_radolan = '{}/{}'.format(s_folder_path, s_data_file_name)
        radolan_data = xr.open_dataset(load_path_radolan, engine='zarr', chunks=None) # chunks = None disables dask overhead
        radolan_data = radolan_data.load()  # loading into RAM

        # DEM
        dem_data = xr.open_dataset(s_dem_path, engine='zarr', chunks=None)
        dem_data = dem_data.load()

        # --- preprocess data ---
        # Radolan
        # Squeeze empty dimension
        radolan_data = radolan_data.squeeze()
        # set all values below 0 to 0
        radolan_data = radolan_data.where(radolan_data >= 0, 0)

        # DEM
        # Only z normalization, no log norm for DEM!
        dem_mean = float(np.mean(dem_data)[s_dem_variable_name].values)
        dem_std = float(np.std(dem_data)[s_dem_variable_name].values)

        # --- xr data and metadata as dict attributes ---
        # The xarray data as attributes
        self.dynamic_data_dict = {
            'radolan': radolan_data
        }

        self.static_data_dict = {
            'dem': dem_data
        }

        # Variable names in xr.Dataset
        self.dynamic_variable_name_dict = {
            'radolan': s_data_variable_name
        }

        self.static_variable_name_dict = {
            'dem': s_dem_variable_name
        }

        # Normalization statistics dict (always logmean, logstd - meaning mean and std of logtransformed data)
        self.dynamic_statistics_dict = {
            'radolan': [radolan_statistics_dict['mean_filtered_log_data'],
                        radolan_statistics_dict['std_filtered_log_data']]
        }

        self.static_statistics_dict = {
            'dem': [dem_mean, dem_std]
        }

    def __getitem__(self, idx):
        sample_coord = self.sample_coords[idx]
        dynamic_samples_dict, static_samples_dict = self.get_sample_from_coords(
            sample_coord,
        )
        return dynamic_samples_dict, static_samples_dict

    def __getitem_with_coord__(self, idx):
        '''
        This is the getitem method for evaluation, where the coordinate is also returned
        Output: (also see get_sample_from_coords())
            dynamic_samples_dict: {'variable_name': timespace chunk that includes input frames and target frame, np.array}
            static_samples_dict: {'variable_name': spacial chunk, np.array}
            sample_coord_float_converted:
                tuple(
                    time: np.timedelta64,
                    y_slice: slice of y coordinates,
                    x_slice slice of x coordinates
                    )
        '''
        sample_coord = self.sample_coords[idx]
        dynamic_samples_dict, static_samples_dict = self.get_sample_from_coords(
            sample_coord,
        )
        sample_coord_float_converted = convert_sample_coord_to_float(sample_coord)
        return dynamic_samples_dict, static_samples_dict, sample_coord_float_converted

    def __len__(self):
        return len(self.sample_coords)

    def get_sample_from_coords(
            self,
            sample_coord: tuple,
            time_step_precipitation_data_minutes=5,
    ):
        '''
        TODO: For evaluation try and call this method directly from DataSet
        This function takes in the coordinates 'input_coord'
            Each input_coord represents one patch that passed the filter.
            The spatial slices in input_coord have the spatial size of the input + the augmentation padding
            The temporal datetime point gives the time of the target frame (as the filter was applied to the target)
            Therefore to get the inputs we have to go back in time relative to the given time in input_coord (depending on lead time and num_input_frames)

        dynamic_data_dict has the following format: {'variable_name': xr.Dataset, ...} Of all variables that are used for the input
        it includes all data that has a time dimension

        static_data_dict has the same format {'variable_name': xr.Dataset, ...} and includes static all data that does
        not have a time dimension

        Output:
            dynamic_samples_dict: {'variable_name': timespace chunk that includes input frames and target frame, np.array}
            static_samples_dict: {'variable_name': spacial chunk, np.array}
        '''

        num_input_frames = self.settings['s_num_input_time_steps']
        s_num_lead_time_steps = self.settings['s_num_lead_time_steps']

        lead_time = s_num_lead_time_steps
        # TODO: NEXT UP implement dict for static and timeseries data:
        # Data loader gets the loading paths for static / timeseries input data as a dict and then this function gets the xr.Datasets
        # as a dict and the returns a static and a time series dict with the values which is then also returned by the data laoder.

        # Make sure this is an int:
        num_input_frames = int(num_input_frames)
        lead_time = int(lead_time)

        # extract coordinates / coordinat slices
        time, y_slice, x_slice = sample_coord

        # TODO: IS LEAD TIME CORRECT? I had to take input 5min * input frame -1 to not choose 4, how about the lead time?
        # Go back in time to get the time slice of the input
        # in oppose to np or list indexing where the last index is not included, the last index is included when taking the datetime slices,
        # therefore num_input_frames - 1!
        time_start = (time -
                      np.timedelta64(time_step_precipitation_data_minutes * (num_input_frames + lead_time), 'm'))
        time_end = time

        time_slice = slice(time_start, time_end)

        # Load dynamic samples (samples with time dimension)
        dynamic_samples_dict = {}
        for key, dynamic_data in self.dynamic_data_dict.items():
            spacetime_sample = dynamic_data.sel(
                time=time_slice,
                y=y_slice,
                x=x_slice,
            )
            variable_name = self.dynamic_variable_name_dict[key]
            dynamic_sample_values = spacetime_sample[variable_name].values  # spacetime_sample.RV_recalc.values
            dynamic_samples_dict[key] = dynamic_sample_values
            if not np.shape(dynamic_sample_values)[0] == num_input_frames + lead_time + 1:
                raise ValueError('The time dim of the sample values is not as expected, check the slicing')

        # Load static samples:
        static_samples_dict = {}
        for key, static_data in self.static_data_dict.items():
            static_sample = static_data.sel(
                y=y_slice,
                x=x_slice,
            )
            variable_name = self.static_variable_name_dict[key]
            static_sample_values = static_sample[variable_name].values
            static_samples_dict[key] = static_sample_values

        return dynamic_samples_dict, static_samples_dict


def random_crop(spacetime_batches: torch.Tensor, s_width_height, **__):
    '''
    Doing a random crop
    '''
    random_crop = transforms.RandomCrop(size=(s_width_height, s_width_height))
    spacetime_batches_cropped = random_crop(spacetime_batches)

    return spacetime_batches_cropped


def create_and_filter_patches(
        y_target, x_target,
        s_num_input_time_steps,
        s_num_lead_time_steps,
        s_folder_path,
        s_data_file_name,
        s_filter_threshold_mm_rain_each_pixel,
        s_filter_threshold_percentage_pixels,
        s_crop_data_time_span,
        **__
) -> tuple[np.array, xr.Dataset, xr.Dataset]:

    # Loading data into xarray
    load_path = '{}/{}'.format(s_folder_path, s_data_file_name)
    data = xr.open_dataset(load_path, engine='zarr')
    if s_crop_data_time_span is not None:
        start_time, stop_time = np.datetime64(s_crop_data_time_span[0]), np.datetime64(s_crop_data_time_span[1])
        crop_slice = slice(start_time, stop_time)
        data = data.sel(time=crop_slice)

    data = data.squeeze()

    # Set negative values to 0
    # I realized that there are quite a few extremely tiny negative values in the data (WHY?), set those to zero
    data_min = data.min(skipna=True, dim=None).RV_recalc.values
    if data_min < -0.1:
        raise ValueError(f'The min value of the data is {data_min}, which is below the threshold of -0.1')
    data = data.where(data >= 0, 0) #All values that are NOT chosen ( >= 0) are set to 0

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
        # time = 1, # No chunking along time dimension
        side="left",  # "left" means that the blocks are aligned to the left of the input
        boundary="trim")  # boundary="trim" removes the last block if it is too small

    # construct a new data set, where the patches are folded into a new dimension
    patches = coarse.construct(
        # time = ("time_outer", "time_inner"),
        y=("y_outer", "y_inner"),  # Those are the patche indecies / the patch dimesnion, each index pair corresponds to one patch
        x=("x_outer", "x_inner")  # Those are the pixel dimensions of the patches
    )

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

    # get the outer coordinates for all valid blocks (valid_time, valid_x, valid_y)
    # (valid_patches_boo is boolean, np.nonzero returns the indecies of the pixels that are non-zero, thus True)
    # valid_target_indecies_outer = np.array(np.nonzero(valid_patches_boo.RV_recalc.values)).T

    return patches, valid_patches_boo, data, data_shortened


def patch_indecies_to_sample_coords(
        data_shortened,
        valid_target_indecies_outer,
        y_target, x_target,
        y_input, x_input,
        y_input_padding, x_input_padding
) -> np.array(tuple):
    '''
    This functions converts the 'valid_target_indecies_outer' which give the outer indecies with respect to 'patches' to
     global coordinates that refer to 'data_shortened'
     !valid_target_indecies_outer includes time and spatial indecies not coordinates!

    The datetime time index of the input and output indecies always refer to the target frame, which the filtering was done on!

    The patches where the larger input patch exceeds the bounds of data_shortened are dropped!
    Therefore the outputs are shorter than valid_target_indecies_outer

    Returns an array of tuples of valid Patch coordinates

    Tuple of each coordinate:
    (datetime, of target frame
    y_slice,
    x_slice
    Spatial coordinates of the input size + the padding. Refer to the coordinates in the preprocessed data, not lat/lon
    So other data can be loaded with this but has to be in correct CRS and transform
    '''
    y_input_padded = y_input + y_input_padding
    x_input_padded = x_input + x_input_padding

    valid_target_indecies_global = []
    valid_center_indecies_global = []
    valid_input_indecies_global = []
    valid_input_coords = []  # These are defined

    num_inputs_exceeding_bounds = 0

    for (time, y_outer, x_outer) in valid_target_indecies_outer:

        y_global_upper = y_outer * y_target
        x_global_left = x_outer * x_target

        # Calculate the global indecies / slices for the targets
        # TODO I think this can be removed in future
        slice_y_global = slice(y_global_upper, y_global_upper + y_target)
        slice_x_global = slice(x_global_left, x_global_left + x_target)
        target_slices = [time, slice_y_global, slice_x_global]

        # Calculate indecies of the patche's center pixels
        center_y_global = y_global_upper + y_target // 2
        center_x_global = x_global_left + x_target // 2
        global_center_indecies = [time, center_y_global, center_x_global]

        # Calculate the global slices for input
        y_slice_input = slice(center_y_global - (y_input_padded // 2), center_y_global + (y_input_padded // 2))
        x_slice_input = slice(center_x_global - (x_input_padded // 2), center_x_global + (x_input_padded // 2))
        input_slices = [time, y_slice_input, x_slice_input]

        # Check if the larger input exceeds size, if not append the patch indecies / slices to the list
        if (
                y_slice_input.start < 0
                or y_slice_input.stop >= data_shortened.sizes['y']
                or x_slice_input.start < 0
                or x_slice_input.stop >= data_shortened.sizes['x']
        ):
            num_inputs_exceeding_bounds += 1
            continue  # Skips the rest of the code in the current loop iteration if input frame exceeds dataset bounds

        # --- Convert indecies to coordinates ---
        time_datetime = data_shortened.time.isel(time=time).values
        y_coords_input = data_shortened.y.isel(y=y_slice_input).values
        x_coords_input = data_shortened.x.isel(x=x_slice_input).values

        y_coords_slices_input = slice(y_coords_input[0], y_coords_input[-1])
        x_coords_slices_input = slice(x_coords_input[0], x_coords_input[-1])

        input_coords = [time_datetime, y_coords_slices_input, x_coords_slices_input]

        valid_target_indecies_global.append(target_slices)
        valid_center_indecies_global.append(global_center_indecies)
        valid_input_indecies_global.append(input_slices)
        valid_input_coords.append(input_coords)

    print(f'{num_inputs_exceeding_bounds} patches dropped as padded input exceeded spatial data bounds')

    return np.array(valid_input_coords)


def split_training_validation(
        data: DatasetGroupBy,

        s_ratio_training_data,  #  Splitting ratio of the groups, not the samples themselves
) -> tuple[xr.Dataset, xr.Dataset]:
    '''
    This randomly splits DatasetGroupBy objects into the training and validation data
    The splitting ratio is given by: s_ratio_training_data
    ! This splits the grouped dataset (default is daliy groups). This means that the days are splitted according to !
    ! the ratio, but each day can gave a different amount of samples that passed the filter !
    The groupby object already sliced the data into slices of a given length, this randomly mixes and concatenates
    according to the given ration of training and validation

    This function has been tested on dicts (see notebook)
    '''
    # Ensure s_ratio_training_data   is between 0 and 1
    if not 0 <= s_ratio_training_data <= 1:
        raise ValueError("s_ratio_training_data must be between 0 and 1.")

    # Shuffle the dictionary keys
    keys = list(data.groups.keys())
    random.shuffle(keys)

    # Calculate the split index
    split_index = int(len(keys) * s_ratio_training_data)

    # Create the training and validation dictionaries
    training_data = xr.concat([data[key] for key in keys[:split_index]], dim='time')
    validation_data = xr.concat([data[key] for key in keys[split_index:]], dim='time')

    return training_data, validation_data


def calc_statistics_on_valid_batches(
        patches: xr.Dataset,
        valid_patches_boo: xr.Dataset
):
    '''
    Calculates 1st and 2nd moment of all valid batches
    This ignores NaNs, flattens the complete data and assumes log1p = log(x+1) for log moments
    '''
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
        float(mean.RV_recalc.values),
        float(std.RV_recalc.values),
        float(log_mean.RV_recalc.values),
        float(log_std.RV_recalc.values)
    )


def calc_linspace_binning(
        data: xr.Dataset,
        mean_filtered_log_data,
        std_filtered_log_data,

        s_linspace_binning_cut_off_unnormalized,
        s_num_bins_crossentropy,
        **__,
):
    '''
    Creates a linspace binning in normalized space of the data. The bin  values are also normalized.
    Creates the vector that gives linspace binning
    Only includes the left edges of the bins.
    The
    '''
    # Calculate min and max for linspace_binning:
    binning_max_unnormed = float(data.max(dim=None, skipna=True).RV_recalc.values)
    binning_min_unnormed = float(data.min(dim=None, skipna=True).RV_recalc.values)

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


# These two functions are use to pass the sample_coord through the datalaoder and subsequently reconstruct it:
def convert_sample_coord_to_float(sample_coord):
    """
    Converts a sample_coord array containing a numpy.datetime64 and slice objects
    into a flattened float array suitable for PyTorch DataLoader.

    Args:
        sample_coord (np.array): An array containing a datetime64 object and two slice objects.

    Returns:
        np.ndarray: A 1D numpy array of floats, where:
                    - datetime64 is converted to a float timestamp.
                    - slice objects are flattened into their start, stop, and step values (step as NaN if None).
    """
    float_array = []
    for item in sample_coord:
        if isinstance(item, np.datetime64):
            # Convert datetime64 to a float timestamp
            float_array.append(item.astype('float64'))
        elif isinstance(item, slice):
            # Convert slice to a tuple (start, stop, step) and flatten it
            float_array.extend([float(item.start), float(item.stop),
                                float(item.step) if item.step is not None else float('nan')])
    return np.array(float_array, dtype=np.float64)


def convert_float_array_to_sample_coord(float_tensor):
    """
    Converts a batched tensor (from DataLoader) containing flattened float representations of sample_coord
    back into the original array format with a datetime64 and slice objects.

    Args:
        float_tensor (torch.Tensor): A 1D tensor where the first value is a timestamp (float),
                                     followed by flattened slice values (start, stop, step).

    Returns:
        np.ndarray: An array where:
                    - The first element is a numpy.datetime64 (converted from the timestamp).
                    - The second and third elements are slice objects reconstructed from their flattened values.
    """
    float_list = float_tensor.tolist()  # Convert tensor back to list of floats
    # Reconstruct the original sample_coord structure
    datetime_value = np.datetime64(int(float_list[0]), 'ns')  # Convert float timestamp to int for datetime64

    # Extract slice values from the list
    slice1 = slice(float_list[1], float_list[2], None if np.isnan(float_list[3]) else float_list[3])
    slice2 = slice(float_list[4], float_list[5], None if np.isnan(float_list[6]) else float_list[6])

    return np.array([datetime_value, slice1, slice2], dtype=object)






