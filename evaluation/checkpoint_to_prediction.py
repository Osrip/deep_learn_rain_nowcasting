from partd.utils import frame

from load_data_xarray import (
    create_patches,
    all_patches_to_datetime_idx_permuts,
    patch_indecies_to_sample_coords,
    split_data_from_time_keys,
    FilteredDatasetXr
)
from helper.checkpoint_handling import load_from_checkpoint, get_checkpoint_names
from helper.memory_logging import print_ram_usage
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import xarray as xr
import einops
from load_data_xarray import convert_float_tensor_to_datetime64_array, split_data_from_time_keys
import numpy as np
import random

class PredictionsToZarrCallback(pl.Callback):

    def __init__(self,
                 orig_training_data,
                 t0_first_input_frame,
                 linspace_binning_params,
                 lead_times,
                 settings):
        '''
        This callback handles saving of the predictions to zarr.
        -------------------------------------------------------------
        ! All predictions are assigned to the FIRST INPUT time step !
        -------------------------------------------------------------
        This design choice has been made, as the datetimes of the split dataset always refer to the target times
         due to filtering on targets

        Input
            data: xr.Dataset:
                Original training data including full time period (wioth input frames)
            t0_first_input_frame: np.datetime64
                The Datetime of the very beginning of the dataset (before splitting)
                So the very first input time step defines t0

        '''
        super().__init__()
        self.settings = settings
        self.mode = 'sample'

        self.linspace_binning_params = linspace_binning_params
        self.lead_times = lead_times
        self.t0_first_input_frame = t0_first_input_frame
        self.orig_training_data = orig_training_data

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0  # Default to 0 if no data_loader_list is passed to trainer
    ):

        return self.on_predict_batch_end_sample_wise(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)


    def on_predict_batch_end_sample_wise(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int = 0, # Set this to 0 by default if no data_loader_list is passed to trainer
    ):
        """
        The predictions of the batch are sample-wise saved as a .zarr
        This is called after predict_step()
        It can handle several dataloaders

        Input:
            outputs: dict
                All tensors have received an added batch dimension (batch_dim = 0) by data loader (Also the entries of sub-dictionaries)!
                {'pred': torch.Tensor,
                'target: sample_metadata_dict}

                sample_metadata_dict: dict
                    All tensors of this sub-dictionary also received an added batch dim by data loader
                    {'time_points_of_spacetime': torch.Tensor         Has to be converted back to datetime
                    'y': torch.Tensor
                    'x': torch.Tensor}

        """
        # Get settings from NetworkL instance

        print_ram_usage()

        s_target_height_width = self.settings['s_target_height_width']
        s_num_input_time_steps = self.settings['s_num_input_time_steps']
        s_num_lead_time_steps = self.settings['s_num_lead_time_steps']
        prediction_dir = self.settings['s_dirs']['prediction_dir']

        linspace_binning_min, linspace_binning_max, linspace_binning = self.linspace_binning_params

        # Get strings to name zarr
        checkpoint_name = trainer.checkpoint_name
        data_loader_name = trainer.data_loader_names[dataloader_idx]

        zarr_file_name = f'model_predictions_{data_loader_name}_{checkpoint_name}.zarr'
        save_zarr_path = f'{prediction_dir}/{zarr_file_name}'

        # Unpacking outputs -> except for loss they are all batched tensors
        # loss = outputs['loss']
        pred_no_softmax = outputs['pred']
        # target = outputs['target']
        # target_binned = outputs['target_binned']
        sample_metadata_dict = outputs['sample_metadata_dict']

        # Softmax predictions
        pred_softmaxed = torch.nn.Softmax(dim=1)(pred_no_softmax)

        batch_size = pred_softmaxed.shape[0]

        # --- Unpack metadata ---
        # The datalaoder added a batch dimension to all entries of the metadata

        # Convert time from float tensor back to np.datetime64
        time_float_tensor_spacetime_chunk = sample_metadata_dict['time_points_of_spacetime'].detach().cpu()
        # Choose the time point of the FIRST INPUT FRAME out of the spacetime chunk (last entry)
        # -> ! EACH PREDICTION IS ASSIGNED TO THE DATETIME OF FIRST INPUT FRAME !
        time_float_tensor_target = time_float_tensor_spacetime_chunk[:, 0]
        time_datetime64_array_target = convert_float_tensor_to_datetime64_array(time_float_tensor_target)

        y_space_chunk = sample_metadata_dict['y']
        x_space_chunk = sample_metadata_dict['x']

        # Center crop spatial metadata to target size = prediction size
        y_target = self._centercrop_on_last_dim_(y_space_chunk, size=s_target_height_width)
        x_target = self._centercrop_on_last_dim_(x_space_chunk, size=s_target_height_width)

        y_target = y_target.cpu().numpy()
        x_target = x_target.cpu().numpy()

        # batch becomes time, channel becomes bin for our xr dataset
        # pred shape: b c h w = batch bin y x

        # Create a dataset from the batch:

        # Add a lead time dimension for current fixed lead time implementation
        pred_softmaxed = einops.rearrange(pred_softmaxed, 'batch bin y x -> 1 batch bin y x')
        pred_softmaxed = pred_softmaxed.detach().cpu().numpy()

        # Process each sample individually
        for i in range(batch_size):
            pred_i = pred_softmaxed[:, i, :, :, :]  # Choose pred from sample i along batch dim
            # Add empty time dimension that we just removed
            pred_i = einops.rearrange(pred_i, 'lead_time bin y x -> lead_time 1 bin y x')
            time_datetime64_array_target_i = time_datetime64_array_target[i]
            y_target_i = y_target[i, :]
            x_target_i = x_target[i, :]

            ds_i = xr.Dataset(
                data_vars={
                    'ml_predictions': (('lead_time', 'time', 'bin', 'y', 'x'), pred_i),
                    # 'coords': (('time', 'y', 'x'), (time_datetime64_array_target_i, y_target_i, x_target_i)),
                },
                coords={
                    'lead_time': self.lead_times,
                    'time': [time_datetime64_array_target_i],  # Wrap time_value in a list to make it indexable
                    'bin': linspace_binning,
                    'y': y_target_i,
                    'x': x_target_i,
                }
            )

            ds_i['time'].encoding['units'] = f'minutes since {self.t0_first_input_frame}'

            ds_i['time'].encoding['dtype'] = 'float64'

            # # Save sample to disk:
            # if i == 0 and batch_idx == 0:
            #     # --- Initialize the zarr file that we will fill up ---


            # Append to the zarr file
            # TODO: Initialize zarr from training 'data' ds. Pass x, y index slices with region.
            #  Docu: https://docs.xarray.dev/en/latest/generated/xarray.Dataset.to_zarr.html
            ds_i.to_zarr(save_zarr_path, mode='r+', region='auto')
            # Appending manual: https://docs.xarray.dev/en/latest/user-guide/io.html#io-zarr
            # Also look at ds.tozarr() docu!


    def _centercrop_on_last_dim_(self, crop_last_dim_tensor: torch.Tensor, size: int) -> torch.Tensor:
        '''
        Per default torchvisions centercrop crops along h and w. This function adds a placeholder dimension
        to do centercropping only along the last dim (dim=-1)
        '''
        # Unsqueeze to add placeholder dimension
        len_d = crop_last_dim_tensor.shape[-1]
        crop_last_dim_tensor_expanded = einops.repeat(crop_last_dim_tensor, '... d -> ... d_new d', d_new=len_d)
        cropped_expanded = T.CenterCrop(size=size)(crop_last_dim_tensor_expanded)

        # **Equality Check**
        # Compare all values along d_new with the first slice
        is_equal = torch.all(
            cropped_expanded == cropped_expanded[..., 0:1, :],
            dim=-2
        )

        # If not all values are equal, raise an error
        if not torch.all(is_equal):
            raise ValueError("Values across 'd_new' are not equal after the operation.")

        # Reduce the tensor back to the original shape
        cropped_orig_shape = einops.reduce(cropped_expanded, '... d_new d -> ... d', 'mean')
        return cropped_orig_shape


def initialize_empty_prediction_dataset(
        orig_data,
        patches,
        linspace_binning_params,
        lead_times,
        pred_save_zarr_path,

        settings,
        s_num_lead_time_steps,
        s_num_input_time_steps,

        minutes_per_frame=5,

        **__,
):
    '''
    This initializes an empty zarr file for each data split (train, val or test), which is then written into sample-wise
    by the prediction callback
    Input:
        orig_data: xr.Dataset
            the original data that has been used to create train/ val / test
            Already has to be cropped in case s_crop_data_time_span has been used
        patches: xr.Dataset
            Those are the split patches (either train, val or test) from which we will extract the time
            dim to initialize the .zarr with. Patches originates from data_shortened, which includes all TARGET
            datetimes.
    '''
    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params

    # ---- Recalc time stamp on TARGET to time stamp on FIRST INPUT

    time_dim_last_frame = patches.time.values
    # Compute frame_delta
    frame_delta = s_num_lead_time_steps + s_num_input_time_steps
    # Compute total minutes to subtract
    delta_minutes = frame_delta * minutes_per_frame
    # Convert to timedelta64 array
    time_delta = np.timedelta64(delta_minutes, 'm')
    # Subtract from the datetime DataArray
    time_dim_first_frame = time_dim_last_frame - time_delta


    # Initialize empty dataarray (initialized with None, not NaN)
    da_empty = xr.DataArray(
        data=None,
        dims=('lead_time', 'time', 'bin', 'y', 'x'),
        coords={
            'lead_time': lead_times,
            'time': time_dim_first_frame,
            'bin': linspace_binning,
            'y': orig_data.y.values,
            'x': orig_data.x.values,
        },
        name="ml_predictions"
    )

    # Convert to dataset
    ds_empty = da_empty.to_dataset()
    # Save
    ds_empty.to_zarr(pred_save_zarr_path, mode='w')


def predict_and_save_to_zarr(
        model,
        patches_train, patches_val, patches_test,
        splits_to_predict_on,
        data_loader_dict,
        checkpoint_name,
        linspace_binning_params,
        lead_times,
        t0_first_input_frame_old, # TODO get rid of this as this is calculated directly in function

        ckp_settings,
        s_folder_path,
        s_data_file_name,
        s_crop_data_time_span,
        s_dirs,
        **__,
):
    prediction_dir = s_dirs['prediction_dir']

    load_path_orig_data = '{}/{}'.format(s_folder_path, s_data_file_name)

    # Load original training data and crop it if that setting was active during training
    orig_data = xr.open_dataset(load_path_orig_data, engine='zarr')
    if s_crop_data_time_span is not None:
        start_time, stop_time = np.datetime64(s_crop_data_time_span[0]), np.datetime64(s_crop_data_time_span[1])
        crop_slice = slice(start_time, stop_time)
        orig_data = orig_data.sel(time=crop_slice)

    # Get t of first input frame of original data (train, val, test) after potentially cutting time span
    t0_first_input_frame = orig_data.time[0].values

    data_loader_list = [data_loader_dict[key] for key in splits_to_predict_on]

    predictions_to_zarr_callback = PredictionsToZarrCallback(
            orig_data,
            t0_first_input_frame,
            linspace_binning_params,
            lead_times,
            ckp_settings
    )

    trainer = pl.Trainer(
        callbacks=predictions_to_zarr_callback,
    )

    trainer.data_loader_names = splits_to_predict_on
    trainer.checkpoint_name = checkpoint_name

    # We need to pass the split patches to initialize the time dim of the zarr
    pacthes_dict = {'train': patches_train,
                      'val': patches_val,
                      'test': patches_test}

    # Initialize empty prediction datasets
    for data_loader_name in splits_to_predict_on:
        pred_zarr_file_name = f'model_predictions_{data_loader_name}_{checkpoint_name}.zarr'
        pred_save_zarr_path = f'{prediction_dir}/{pred_zarr_file_name}'

        patches = pacthes_dict[data_loader_name]

        initialize_empty_prediction_dataset(
            orig_data,
            patches,
            linspace_binning_params,
            lead_times,
            pred_save_zarr_path,
            ckp_settings,
            **ckp_settings
        )

    # trainer.predict already does torch.no_grad() and calls model.eval(), (according to o1), so no need for extras here
    # https://lightning.ai/docs/pytorch/stable/deploy/production_basic.html

    trainer.predict(
        model=model,
        dataloaders=data_loader_list,
        return_predictions=False  # By default, lightning aggregates the output of all batches, disable this to prevent memory overflow
    )


def sample_coords_for_all_patches(
        train_time_keys, val_time_keys, test_time_keys,
        max_num_frames_per_split,

        settings,
        s_target_height_width,
        s_input_height_width,
        s_split_chunk_duration,
        **__,
):
    """
    This creates returns the sample_coords for all, unfiltered patches.

    Input
        train_time_keys, val_time_keys, test_time_keys: list(np.datetime64)
            These are the time keys that determine the train, val and test sets
            They refer to the group names of patches.resample(time=s_split_chunk_duration)

        max_num_frames: int / None
            maximal number of frames that shall be predicted on each split. If None the whole split is predicted

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
    y_target, x_target = s_target_height_width, s_target_height_width  # 73, 137 # how many pixels in y and x direction
    y_input, x_input = s_input_height_width, s_input_height_width
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
        # 'back in time' to go frame target time to input time.
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


    # --- Subsample patches along time dim if necessary ---
    if max_num_frames_per_split is not None:
        patches_dict = {'train': patches_train, 'val': patches_val, 'test': patches_test}

        for split_name, patches in patches_dict.items():
            if len(patches.time.values) > max_num_frames_per_split:
                subsampled_patch_time_dim = np.random.choice(
                    patches.time.values,
                    size=max_num_frames_per_split,
                    replace=False
                )
                subsampled_patches = patches.sel(time=subsampled_patch_time_dim)
                patches_dict[split_name] = subsampled_patches

        patches_train, patches_val, patches_test = patches_dict['train'], patches_dict['val'], patches_dict['test']

    # --- From patches create sample coords ---

    # As we want all patches and do not do any filtering in this case we simply permute the _outer patch indecies

    time_dim_name, y_dim_name, x_dim_name = [
        'time',
        'y_outer',
        'x_outer',
    ]

    # Get index permutations [[time: np.datetime64, y_idx: int, x_idx: int], ...] for all patches
    train_datetime_idx_permuts = all_patches_to_datetime_idx_permuts(patches_train, time_dim_name, y_dim_name, x_dim_name)
    val_datetime_idx_permuts = all_patches_to_datetime_idx_permuts(patches_val, time_dim_name, y_dim_name, x_dim_name)
    test_datetime_idx_permuts = all_patches_to_datetime_idx_permuts(patches_test, time_dim_name, y_dim_name, x_dim_name)

    # --- Check for duplicates ---
    # Combine all sample coordinates
    all_sample_coords = train_datetime_idx_permuts + val_datetime_idx_permuts + test_datetime_idx_permuts

    # Calculate the total number of samples and the number of unique samples
    total_samples = len(all_sample_coords)
    unique_samples = len(set(all_sample_coords))

    # Check for duplicates
    if total_samples != unique_samples:
        num_duplicates = total_samples - unique_samples
        raise ValueError(
            f'There are {num_duplicates} duplicate samples among train, val, and test sets.'
        )

    # ... and calculate the sample coords with respect to the CRS and projection of data_shortened of them
    train_sample_coords = patch_indecies_to_sample_coords(
        data_shortened,
        train_datetime_idx_permuts,
        y_target, x_target,
        y_input, x_input,
        y_input_padding, x_input_padding,
    )

    val_sample_coords = patch_indecies_to_sample_coords(
        data_shortened,
        val_datetime_idx_permuts,
        y_target, x_target,
        y_input, x_input,
        y_input_padding, x_input_padding,
    )

    test_sample_coords = patch_indecies_to_sample_coords(
        data_shortened,
        test_datetime_idx_permuts,
        y_target, x_target,
        y_input, x_input,
        y_input_padding, x_input_padding,
    )

    # TODO: get rid of this, this will be calculated directly from data in future
    t0_first_input_frame = data.time[0].values

    return (
        train_sample_coords, val_sample_coords, test_sample_coords,
        patches_train, patches_val, patches_test,
        t0_first_input_frame
    )


def create_predict_dataloaders(
        train_sample_coords, val_sample_coords, test_sample_coords,
        radolan_statistics_dict,

        settings,
        s_batch_size,
        s_num_workers_data_loader,
        **__,
):

    # --- Create Datasets ---
    train_data_set_eval = FilteredDatasetXr(
        train_sample_coords,
        radolan_statistics_dict,
        mode='predict',
        settings=settings,
    )

    val_data_set_eval = FilteredDatasetXr(
        val_sample_coords,
        radolan_statistics_dict,
        mode='predict',
        settings=settings,
    )

    test_data_set_eval = FilteredDatasetXr(
        test_sample_coords,
        radolan_statistics_dict,
        mode='predict',
        settings=settings,
    )

    # --- Create Dataloaders ---

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
        linspace_binning_params,
        splits_to_predict_on,

        ckp_settings,  # Make sure to pass the settings of the checkpoint
        s_dirs,

        max_num_frames_per_split=None,

        **__,
):
    """
    This creates a .zarr file for all predictions of the model checkpoint

    !Caution!
        This assumes that the data that you are predicting with has exactly the same length through s_crop_data_time_span
        as in the training setting! Do not simply change the time_key lengths if you want to predict less!

    Input
        train_time_keys, val_time_keys, test_time_keys: list(np.datetime64)
            These are the time keys that determine the train, val and test sets
            They refer to the group names of patches.resample(time=s_split_chunk_duration)

        splits_to_predict_on: List of str
            List of strings containing at least one of
            ['train', 'val', 'test']

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

        max_num_frames: int / None
            maximal number of frames that shall be predicted on each split. If None the whole split is predicted
    """

    print('Predicting on unfiltered patches')

    if not any(item in ['train', 'val', 'test'] for item in splits_to_predict_on):
        raise ValueError("splits_to_predict_on must be a list of strings containing at least one of"
                         " ['train', 'val', 'test']")



    # For now, with fixed lead time simply create lead times like this:
    lead_times = [ckp_settings['s_num_lead_time_steps']]
    save_dir = s_dirs['save_dir']

    checkpoint_names = get_checkpoint_names(save_dir)

    # Only do prediction for last checkpoint
    checkpoint_name_to_predict = [name for name in checkpoint_names if 'last' in name][0]

    # Get the sample coords for all -unfiltered- patches
    (
        train_sample_coords, val_sample_coords, test_sample_coords,
        patches_train, patches_val, patches_test,  # We need the patches later on to initialize time dim of pred zarr
        t0_first_input_frame,
     ) = sample_coords_for_all_patches(
        train_time_keys, val_time_keys, test_time_keys,
        max_num_frames_per_split,
        ckp_settings,
        **ckp_settings,
    )

    # Create data loaders
    train_data_loader_predict, val_data_loader_predict, test_data_loader_predict = create_predict_dataloaders(
        train_sample_coords, val_sample_coords, test_sample_coords,
        radolan_statistics_dict,
        ckp_settings,
        **ckp_settings,
    )

    model = load_from_checkpoint(
        save_dir,
        checkpoint_name_to_predict,

        ckp_settings,
        **ckp_settings,
    )

    data_loader_dict = {'train': train_data_loader_predict,
                        'val': val_data_loader_predict,
                        'test': test_data_loader_predict}

    predict_and_save_to_zarr(
        model,
        patches_train, patches_val, patches_test,
        splits_to_predict_on,
        data_loader_dict,
        checkpoint_name_to_predict,
        linspace_binning_params,
        lead_times,
        t0_first_input_frame,
        ckp_settings,
        **ckp_settings,
    )

















