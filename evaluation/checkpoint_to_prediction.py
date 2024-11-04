from load_data_xarray import (
    create_patches,
    all_patches_to_datetime_idx_permuts,
    patch_indecies_to_sample_coords,
    split_data_from_time_keys,
    FilteredDatasetXr
)
from helper.checkpoint_handling import load_from_checkpoint, get_checkpoint_names
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import xarray as xr
import einops
from load_data_xarray import convert_float_tensor_to_datetime64_array, split_data_from_time_keys
import numpy as np

class PredictionsToZarrCallback(pl.Callback):

    def __init__(self,
                 orig_training_data,
                 t0_first_input_frame,
                 linspace_binning_params,
                 lead_times,
                 settings):
        '''
        This callback handles saving of the predictions to zarr.
        All predictions are assigned to the TARGET time step!
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
        # Dynamically call the appropriate method based on the mode
        if self.mode == "sample":
            return self.on_predict_batch_end_sample_wise(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        else:
            return self.on_predict_batch_end_batch_wise(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

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
            Outputs: dict
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
        pred = outputs['pred']
        # target = outputs['target']
        # target_binned = outputs['target_binned']
        sample_metadata_dict = outputs['sample_metadata_dict']

        batch_size = pred.shape[0]

        # --- Unpack metadata ---
        # The datalaoder added a batch dimension to all entries of the metadata

        # Convert time from float tensor back to np.datetime64
        time_float_tensor_spacetime_chunk = sample_metadata_dict['time_points_of_spacetime'].detach().cpu()
        # Choose the time point of the FIRST INPUT FRAME out of the spacetime chunk (last entry)
        # -> EACH PREDICTION IS ASSIGNED TO THE DATETIME OF TARGET
        time_float_tensor_target = time_float_tensor_spacetime_chunk[:, -1]
        time_datetime64_array_target = convert_float_tensor_to_datetime64_array(time_float_tensor_target)

        y_space_chunk = sample_metadata_dict['y']
        x_space_chunk = sample_metadata_dict['x']

        # Centercrop spatial metadata to target size = prediction size
        y_target = self._centercrop_on_last_dim_(y_space_chunk, size=s_target_height_width)
        x_target = self._centercrop_on_last_dim_(x_space_chunk, size=s_target_height_width)

        y_target = y_target.cpu().numpy()
        x_target = x_target.cpu().numpy()

        # batch becomes time, channel becomes bin for our xr dataset
        # pred shape: b c h w = batch bin y x

        # Create a dataset from the batch:

        # Add a lead time dimension for current fixed lead time implementation
        pred = einops.rearrange(pred, 'batch bin y x -> 1 batch bin y x')
        pred = pred.detach().cpu().numpy()

        # Process each sample individually
        for i in range(batch_size):
            pred_i = pred[:, i, :, :, :]  # Choose pred from sample i along batch dim
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
                    'bin': linspace_binning,
                    'time': [time_datetime64_array_target_i],  # Wrap time_value in a list to make it indexable
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
            print(f'Save sample num {i} of batch {batch_idx}')


    # def on_predict_batch_end_sample_wise(
    #         self,
    #         trainer: "pl.Trainer",
    #         pl_module: "pl.LightningModule",
    #         outputs,
    #         batch,
    #         batch_idx: int,
    #         dataloader_idx: int = 0, # Set this to 0 by default if no data_loader_list is passed to trainer
    # ):
    #     """
    #     The predictions of the batch are sample-wise saved as a .zarr
    #     This is called after predict_step()
    #     It can handle several dataloaders
    #
    #     Input:
    #         Outputs: dict
    #             All tensors have received an added batch dimension (batch_dim = 0) by data loader (Also the entries of sub-dictionaries)!
    #             {'pred': torch.Tensor,
    #             'target: sample_metadata_dict}
    #
    #             sample_metadata_dict: dict
    #                 All tensors of this sub-dictionary also received an added batch dim by data loader
    #                 {'time_points_of_spacetime': torch.Tensor         Has to be converted back to datetime
    #                 'y': torch.Tensor
    #                 'x': torch.Tensor}
    #
    #     """
    #     # Get settings from NetworkL instance
    #
    #     s_target_height_width = self.settings['s_target_height_width']
    #     s_num_input_time_steps = self.settings['s_num_input_time_steps']
    #     s_num_lead_time_steps = self.settings['s_num_lead_time_steps']
    #     prediction_dir = self.settings['s_dirs']['prediction_dir']
    #
    #     # Get strings to name zarr
    #     checkpoint_name = trainer.checkpoint_name
    #     data_loader_name = trainer.data_loader_names[dataloader_idx]
    #
    #     zarr_file_name = f'model_predictions_{data_loader_name}_{checkpoint_name}.zarr'
    #     save_zarr_path = f'{prediction_dir}/{zarr_file_name}'
    #
    #     # Unpacking outputs -> except for loss they are all batched tensors
    #     # loss = outputs['loss']
    #     pred = outputs['pred']
    #     # target = outputs['target']
    #     # target_binned = outputs['target_binned']
    #     sample_metadata_dict = outputs['sample_metadata_dict']
    #
    #     batch_size = pred.shape[0]
    #
    #     # --- Unpack metadata ---
    #     # The datalaoder added a batch dimension to all entries of the metadata
    #
    #     # Convert time from float tensor back to np.datetime64
    #     time_float_tensor_spacetime_chunk = sample_metadata_dict['time_points_of_spacetime'].detach().cpu()
    #     # Choose the time point of the FIRST INPUT FRAME out of the spacetime chunk (last entry)
    #     # -> EACH PREDICTION IS ASSIGNED TO THE DATETIME OF FIRST INPUT FRAME
    #     time_float_tensor_target = time_float_tensor_spacetime_chunk[:, 0]
    #     time_datetime64_array_target = convert_float_tensor_to_datetime64_array(time_float_tensor_target)
    #
    #     y_space_chunk = sample_metadata_dict['y']
    #     x_space_chunk = sample_metadata_dict['x']
    #
    #     # Centercrop spatial metadata to target size = prediction size
    #     y_target = self._centercrop_on_last_dim_(y_space_chunk, size=s_target_height_width)
    #     x_target = self._centercrop_on_last_dim_(x_space_chunk, size=s_target_height_width)
    #
    #     y_target = y_target.cpu().numpy()
    #     x_target = x_target.cpu().numpy()
    #
    #     # --- Unchanging metadata ---
    #     linspace_binning_params = trainer.linspace_binning_params
    #     linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params
    #
    #     lead_times = trainer.lead_times
    #
    #     # batch becomes time, channel becomes bin for our xr dataset
    #     # pred shape: b c h w = batch bin y x
    #
    #     # Create a dataset from the batch:
    #
    #     # Add a lead time dimension for current fixed lead time implementation
    #     pred = einops.rearrange(pred, 'batch bin y x -> 1 batch bin y x')
    #     pred = pred.detach().cpu().numpy()
    #
    #     # Process each sample individually
    #     for i in range(batch_size):
    #         pred_i = pred[:, i, :, :, :]  # Choose pred from sample i along batch dim
    #         # Add empty time dimension that we just removed
    #         pred_i = einops.rearrange(pred_i, 'lead_time bin y x -> lead_time 1 bin y x')
    #         time_datetime64_array_target_i = time_datetime64_array_target[i]
    #         y_target_i = y_target[i, :]
    #         x_target_i = x_target[i, :]
    #
    #         ds_i = xr.Dataset(
    #             data_vars={
    #                 'ml_predictions': (('lead_time', 'time', 'bin', 'y', 'x'), pred_i),
    #                 # 'coords': (('time', 'y', 'x'), (time_datetime64_array_target_i, y_target_i, x_target_i)),
    #             },
    #             coords={
    #                 'lead_time': lead_times,
    #                 'bin': linspace_binning,
    #                 'time': [time_datetime64_array_target_i],  # Wrap time_value in a list to make it indexable
    #                 'y': y_target_i,
    #                 'x': x_target_i,
    #             }
    #         )
    #
    #         ds_i['time'].encoding['units'] = f'minutes since {self.t0_first_input_frame}'
    #
    #         ds_i['time'].encoding['dtype'] = 'float64'
    #
    #         # Save sample to disk:
    #         if i == 0 and batch_idx == 0:
    #             # --- Initialize the zarr file that we will fill up ---
    #
    #             # As we are assigning each prediction to the first input frame, we cut off the end of the training data
    #             # We use this to initialize the prediction zarr file
    #                 slice_cut_off_end = slice(0, - (s_num_input_time_steps + s_num_lead_time_steps))
    #                 training_data_cut_end = self.orig_training_data.isel(time=slice_cut_off_end)
    #                 # Initialize the zarr file
    #                 nan_ds = xr.full_like(training_data_cut_end, fill_value=np.nan)
    #                 # Drop the step diemnsion which is a legacy that used to include predicrtions
    #                 nan_ds = nan_ds.squeeze()
    #                 nan_ds = nan_ds.drop_vars('step')
    #                 # Add lead_time dim. If len(lead_times) > 1 data will be broadcasted / copied to fill new entries,
    #                 # which are nans anyways
    #                 nan_ds = nan_ds.expand_dims({'lead_time': lead_times})
    #                 # Add bin dim:
    #                 nan_ds = nan_ds.expand_dims({'bin': linspace_binning})
    #                 #Rename:
    #                 nan_ds = nan_ds.rename({'RV_recalc': 'ml_predictions'})
    #                 # Transpose all data variables in the dataset to the specified order
    #                 nan_ds = nan_ds.transpose('lead_time', 'time', 'bin', 'y', 'x')
    #                 nan_ds.to_zarr(save_zarr_path, mode='w')
    #
    #
    #         # Append to the zarr file
    #         # TODO: Initialize zarr from training 'data' ds. Pass x, y index slices with region.
    #         #  Docu: https://docs.xarray.dev/en/latest/generated/xarray.Dataset.to_zarr.html
    #         ds_i.to_zarr(save_zarr_path, mode='r+', region='auto')
    #         print(f'Save sample num {i} of batch {batch_idx}')


    # def on_predict_batch_end_batch_wise(
    #         self,
    #         trainer: "pl.Trainer",
    #         pl_module: "pl.LightningModule",
    #         outputs,
    #         batch,
    #         batch_idx: int,
    #         dataloader_idx: int = 0,  # Set this to 0 by default if no data_loader_list is passed to trainer
    # ):
    #     """
    #     This is called after predict_step()
    #     """
    #     # Get settings from NetworkL instance
    #     settings = trainer.lightning_module.settings
    #     s_target_height_width = settings['s_target_height_width']
    #
    #     # Get strings to name zarr
    #     checkpoint_name = trainer.checkpoint_name
    #     data_loader_name = trainer.data_loader_names[dataloader_idx]
    #
    #     zarr_file_name = f'model_predictions_{data_loader_name}_{checkpoint_name}.zarr'
    #     # TODO get path to save zarr in (save this globally instead of run folder?
    #     save_zarr_path = zarr_file_name
    #     # TODO get t0 of radolan
    #     t0_of_radolan = None
    #
    #     # Unpacking outputs -> except for loss they are all batched tensors
    #     loss = outputs['loss']
    #     pred = outputs['pred']
    #     target = outputs['target']
    #     target_binned = outputs['target_binned']
    #     sample_metadata_dict = outputs['sample_metadata_dict']
    #
    #     # --- Unpack metadata ---
    #     # The datalaoder added a batch dimension to all entries of the metadata
    #
    #     # Convert time from float tensor back to np.datetime64
    #     time_float_tensor_spacetime_chunk = sample_metadata_dict['time_points_of_spacetime'].detach().cpu()
    #     # Choose the time point of the target out of the spacetime chunk (last entry)
    #     time_float_tensor_target = time_float_tensor_spacetime_chunk[:, -1]
    #     time_datetime64_array_target = convert_float_tensor_to_datetime64_array(time_float_tensor_target)
    #
    #     y_space_chunk = sample_metadata_dict['y']
    #     x_space_chunk = sample_metadata_dict['x']
    #
    #     # Centercrop spatial metadata to target size = prediction size
    #     y_target = self._centercrop_on_last_dim_(y_space_chunk, size=s_target_height_width)
    #     x_target = self._centercrop_on_last_dim_(x_space_chunk, size=s_target_height_width)
    #
    #     y_target = y_target.cpu().numpy()
    #     x_target = x_target.cpu().numpy()
    #
    #     # --- Unchanging metadata ---
    #     linspace_binning_params = trainer.linspace_binning_params
    #     linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params
    #
    #     lead_times = trainer.lead_times
    #
    #     # batch becomes time, channel becomes bin for our xr dataset
    #     # pred shape: b c h w = batch bin y x
    #
    #     # Create a dataset from the batch:
    #
    #     # Add a lead time dimension for current fixed lead time implementation
    #     pred = einops.rearrange(pred, 'batch bin y x -> 1 batch bin y x')
    #     # pred = pred.unsqueeze(0)
    #     pred = pred.detach().cpu().numpy()
    #
    #     batch_size = pred.shape[0]
    #
    #     # ds_i['time'].encoding['units'] = f'minutes since {t0_of_radolan}'
    #     #
    #     # ds_i['time'].encoding['dtype'] = 'float64'
    #
    #     batch_ds = xr.Dataset(
    #         data_vars={
    #             'ml_predictions': (('lead_time', 'time', 'bin', 'y', 'x'), pred)
    #         },
    #         coords={
    #             'lead_time': lead_times,
    #             'bin': linspace_binning,
    #             'time': time_datetime64_array_target,  # Wrap time_value in a list to make it indexable
    #             'y': (('time', 'y'), y_target),
    #             'x': (('time', 'x'), x_target),
    #         }
    #     )
    #     # It looks like this is not possible:
    #     # xarray.core.merge.MergeError: coordinate y shares a name with a dataset dimension,
    #     # but is not a 1D variable along that dimension. This is disallowed by the xarray data model.
    #
    #     # Save sample to disk:
    #     if batch_idx == 0:
    #         # Initialize the zarr file
    #         batch_ds.to_zarr(save_zarr_path, mode='w')
    #     else:
    #         # Append to the zarr file
    #         # TODO: We are really appending in time, y and x ... how do I do that with arg append_dim= ?
    #         batch_ds.to_zarr(save_zarr_path, mode='a-', append_dim='time')

        # # Appending manual: https://docs.xarray.dev/en/latest/user-guide/io.html#io-zarr

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
        time_keys,
        linspace_binning_params,
        lead_times,
        pred_save_zarr_path,
        settings,
        s_split_chunk_duration,
        **__,
):
    '''
    Input:
        orig_data: xr.Dataset
            the original data that has been used to create train/ val / test
            Already has to be cropped in case s_crop_data_time_span has been used
    '''
    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params

    # From the original data we are creating the original split (train, val or test) in order to
    # receive the original time dimension along which we will fill up the data
    orig_data_resampled = orig_data.resample(time=s_split_chunk_duration)
    split_data = split_data_from_time_keys(orig_data_resampled, time_keys)

    # Initialize empty dataarray (initialized with None, not NaN)
    da_empty = xr.DataArray(
        data=None,
        dims=('lead_time', 'time', 'bin', 'y', 'x'),
        coords={
            'lead_time': lead_times,
            'bin': linspace_binning,
            'time': split_data.time,  # TODO
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
        train_time_keys, val_time_keys, test_time_keys,
        data_to_predict_on,
        data_loader_dict,
        checkpoint_name,
        linspace_binning_params,
        lead_times,
        t0_first_input_frame_old, # TODO get rid of this as this ios calculated directly in function

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

    data_loader_list = [data_loader_dict[key] for key in data_to_predict_on]

    trainer = pl.Trainer(
        callbacks=PredictionsToZarrCallback(
            orig_data,
            t0_first_input_frame,
            linspace_binning_params,
            lead_times,
            ckp_settings)
    )

    trainer.data_loader_names = data_to_predict_on
    trainer.checkpoint_name = checkpoint_name

    time_keys_dict = {'train': train_time_keys,
                      'val': val_time_keys,
                      'test': test_time_keys}

    # Initialize empty prediction datasets
    for data_loader_name in data_to_predict_on:
        pred_zarr_file_name = f'model_predictions_{data_loader_name}_{checkpoint_name}.zarr'
        pred_save_zarr_path = f'{prediction_dir}/{pred_zarr_file_name}'

        time_keys = time_keys_dict[data_loader_name]

        initialize_empty_prediction_dataset(
            orig_data,
            time_keys,
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
    )


def sample_coords_for_all_patches(
        train_time_keys, val_time_keys, test_time_keys,

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
            f'There are {num_duplicates} duplicates among train, val, and test sets.'
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

    return train_sample_coords, val_sample_coords, test_sample_coords, t0_first_input_frame


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
        data_to_predict_on,

        ckp_settings,  # Make sure to pass the settings of the checkpoint
        s_dirs,
        **__,
):
    """
    This creates a .zarr file for all predictions of the model checkpoint

    Input
        train_time_keys, val_time_keys, test_time_keys: list(np.datetime64)
            These are the time keys that determine the train, val and test sets
            They refer to the group names of patches.resample(time=s_split_chunk_duration)
        data_to_predict_on: List of str
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
    """

    if not any(item in ['train', 'val', 'test'] for item in data_to_predict_on):
        raise ValueError("data_to_predict_on must be a list of strings containing at least one of"
                         " ['train', 'val', 'test']")

    # For now, with fixed lead time simply create lead times like this:
    lead_times = [ckp_settings['s_num_lead_time_steps']]
    save_dir = s_dirs['save_dir']

    # TODO: Implement taking a subset of train_time_keys, val_time_keys, test_time_keys,
    #  to save resources with predictions? Maybe just a date range?

    checkpoint_names = get_checkpoint_names(save_dir)

    # Only do prediction for last checkpoint
    checkpoint_name_to_predict = [name for name in checkpoint_names if 'last' in name][0]

    # Get the sample coords for all -unfiltered- patches
    train_sample_coords, val_sample_coords, test_sample_coords, t0_first_input_frame = sample_coords_for_all_patches(
        train_time_keys, val_time_keys, test_time_keys,
        ckp_settings,
        **ckp_settings,
    )

    # Create dataloaders
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
        train_time_keys, val_time_keys, test_time_keys,
        data_to_predict_on,
        data_loader_dict,
        checkpoint_name_to_predict,
        linspace_binning_params,
        lead_times,
        t0_first_input_frame,
        ckp_settings,
        **ckp_settings,
    )

    # TODO: Predictions, Patch assembly, chunk-wise saving to zarr

    #Todo somehow do predictions and chunk them in time
    # such that we can iteratively write zarr to disk and dont overload RAM

    # TODO: initialize NetworkL and lightning trainer
    # Or maybe it is better not to use lightning in this case?


    # define a def predict function in NetworkL

    # load data with __getitem_evaluation__

    # # Perform predictions
    # predictions_train = trainer.predict(model, dataloaders=train_data_loader_eval)
    # predictions_val = trainer.predict(model, dataloaders=val_data_loader_eval)
    # predictions_test = trainer.predict(model, dataloaders=test_data_loader_eval)
















