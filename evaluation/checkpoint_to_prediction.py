from load_data_xarray import (
    create_patches,
    all_patches_to_datetime_idx_permuts,
    patch_indecies_to_sample_coords,
    split_data_from_time_keys,
    FilteredDatasetXr
)
from helper.checkpoint_handling import load_from_checkpoint, get_checkpoint_names

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import xarray as xr
import einops
from load_data_xarray import convert_float_tensor_to_datetime64_array

class PredictionsToZarrCallback(pl.Callback):

    def __init__(self):
        super().__init__()

    def on_predict_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int = 0, # Set this to 0 by default if no data_loader_list is passed to trainer
    ):
        """
        This is called after predict_step()
        """
        # Get strings to name zarr
        checkpoint_name = trainer.checkpoint_name
        data_loader_name = trainer.data_loader_names[dataloader_idx]

        zarr_file_name = f'model_predictions_{data_loader_name}_{checkpoint_name}.zarr'
        # TODO get path to save zarr in (save this globally instead of run folder?
        save_zarr_path = zarr_file_name
        # TODO get t0 of radolan
        t0_of_radolan = None


        # Unpacking outputs -> except for loss they are all batched tensors
        loss = outputs['loss']
        pred = outputs['pred']
        target = outputs['target']
        target_binned = outputs['target_binned']
        sample_metadata_dict = outputs['sample_metadata_dict']

        # --- Unpack metadata ---

        # Convert time from float tensor back to np.datetime64
        time_float_tensor_spacetime_chunk = sample_metadata_dict['time_points_of_spacetime'].detach().cpu()
        # Choose the time point of the target out of the spacetime chunk (last entry)
        time_float_tensor_target = time_float_tensor_spacetime_chunk[:, -1]
        time_datetime64_array_target = convert_float_tensor_to_datetime64_array(time_float_tensor_target)

        y = sample_metadata_dict['y'].detach().cpu().numpy()
        x = sample_metadata_dict['x'].detach().cpu().numpy()

        # --- Unchanging metadata ---
        linspace_binning_params = trainer.linspace_binning_params
        linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params

        lead_times = trainer.lead_times

        # batch becomes time, channel becomes bin for our xr dataset
        # pred shape: b c h w = batch bin y x

        # Create a dataset from the batch:

        # Add a lead time dimension for current fixed lead time implementation
        pred = einops.rearrange(pred, 'batch bin y x -> 1 batch bin y x')
        # pred = pred.unsqueeze(0)
        pred = pred.detach().cpu().numpy()

        batch_size = pred.shape[0]

        # Process each sample individually
        for i in range(batch_size):
            pred_i = pred[:, i, :, :, :]  # Choose pred from sample i along batch dim
            # Add empty time dimension that we just removed
            pred_i = einops.rearrange(pred_i, 'lead_time bin y x -> lead_time 1 bin y x')
            time_datetime64_array_target_i = time_datetime64_array_target[i]
            y_i = y[:, i]
            x_i = x[:, i]

            ds_i = xr.Dataset(
                data_vars={
                    'ml_predictions': (('lead_time', 'time', 'bin', 'y', 'x'), pred_i)
                },
                coords={
                    'lead_time': lead_times,
                    'bin': linspace_binning,
                    'time': time_datetime64_array_target_i,  # Wrap time_value in a list to make it indexable
                    'y': y_i,
                    'x': x_i,
                }
            )

            ds_i['time'].encoding['units'] = f'minutes since {t0_of_radolan}'

            ds_i['time'].encoding['dtype'] = 'float64'

            # Save sample to disk:
            if i == 0 and batch_idx == 0:
                # Initialize the zarr file
                ds_i.to_zarr(save_zarr_path, mode='w')
            else:
                # Append to the zarr file
                #TODO: We are really appending in time, y and x ... how do I do that with arg append_dim= ?
                ds_i.to_zarr(ds_i['save_zarr_path'], mode='a-', append_dim='time')
        #


        # TODO Seems not to be possible to save the whole batch at once ... or maybe?
        #  Error:
        # xarray.core.variable.MissingDimensionsError: cannot set variable 'y' with 2-dimensional data without explicit dimension names. Pass a tuple of (dims, data) instead.
        radolan_pred_ds = xr.Dataset(
            data_vars={
                'ml_predictions': (('lead_time', 'time', 'bin', 'y', 'x'), pred)
            },
            coords={
                'lead_time': lead_times,
                'bin': linspace_binning,
                'time': time_datetime64_array_target,  # Wrap time_value in a list to make it indexable
                'y': y,
                'x': x,
            }
        )





        if batch_idx == 0:
            pass




        #  TODO save zarr batch wise

        # # Set encoding for the time coordinate
        #
        # radolan_pred_ds['time'].encoding['units'] = f'minutes since {t0_of_radolan}'
        #
        # radolan_pred_ds['time'].encoding['dtype'] = 'float64'
        #
        # # This way we are appending on disk via time dimension
        # if i == 0:
        #     radolan_pred_ds.to_zarr(pre_settings['save_zarr_path'], mode='w')
        # else:
        #     radolan_pred_ds.to_zarr(pre_settings['save_zarr_path'], mode='a-', append_dim='time')
        #
        # # Appending manual: https://docs.xarray.dev/en/latest/user-guide/io.html#io-zarr

def predict_and_save_to_zarr(
        model,
        data_loader_dict,
        checkpoint_name,
        linspace_binning_params,
        lead_times
):
    data_loader_list = [data_loader_dict[key] for key in data_loader_dict.keys()]
    trainer = pl.Trainer(
        callbacks=PredictionsToZarrCallback()
    )

    trainer.data_loader_names = list(data_loader_dict.keys())
    trainer.checkpoint_name = checkpoint_name

    trainer.linspace_binning_params =linspace_binning_params
    trainer.lead_times = lead_times

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

    return train_sample_coords, val_sample_coords, test_sample_coords


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

    # For now, with fixed lead time simply create lead times like this:
    lead_times = [ckp_settings['s_num_lead_time_steps']]
    save_dir = s_dirs['save_dir']

    # TODO: Implement taking a subset of train_time_keys, val_time_keys, test_time_keys,
    #  to save resources with predictions? Maybe just a date range?

    checkpoint_names = get_checkpoint_names(save_dir)
    for checkpoint_name in checkpoint_names:

        # Get the sample coords for all -unfiltered- patches
        train_sample_coords, val_sample_coords, test_sample_coords = sample_coords_for_all_patches(
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
            checkpoint_name,

            ckp_settings,
            **ckp_settings,
        )

        data_loader_dict = {'train': train_data_loader_predict, 'val': val_data_loader_predict}

        predict_and_save_to_zarr(
            model,
            data_loader_dict,
            checkpoint_name,
            linspace_binning_params,
            lead_times,
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
















