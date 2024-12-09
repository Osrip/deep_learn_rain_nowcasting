import torch
from torch.utils.data import DataLoader
import xarray as xr
from load_data_xarray import FilteredDatasetXr
import pytorch_lightning as pl
from helper.helper_functions import center_crop_1d
import torch
import torchvision.transforms as T
from helper.pre_process_target_input import one_hot_to_lognormed_mm, inverse_normalize_data





class EvaluateBaselineCallback(pl.Callback):

    def __init__(
            self,
            linspace_binning_params,
            checkpoint_name,
            samples_have_padding,
            settings,
    ):
        '''
        This callback handles saving of the predictions to zarr.
        -------------------------------------------------------------
        ! All predictions are assigned to the FIRST INPUT time step !
        -------------------------------------------------------------
        This design choice has been made, as the datetimes of the split dataset always refer to the target times
         due to filtering on targets

        Input
            baseline: xr.Dataset:
                Original training data including full time period (wioth input frames)
            t0_first_input_frame: np.datetime64
                The Datetime of the very beginning of the dataset (before splitting)
                So the very first input time step defines t0
            samples_have_padding: bool
                If True this indicates an input padding, therefore we will center crop to s_width_height
        '''

        super().__init__()
        self.settings = settings
        self.linspace_binning_params = linspace_binning_params
        self.checkpoint_name = checkpoint_name
        self.samples_have_padding = samples_have_padding

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0  # Default to 0 if no data_loader_list is passed to trainer
    ):
        """
        Input:
            outputs: dict
                All tensors have received an added batch dimension (batch_dim = 0) by data loader (Also the entries of sub-dictionaries)!
                {'pred': torch.Tensor,
                'target: sample_metadata_dict}

                baseline: dict
                    All tensors of this sub-dictionary also received an added batch dim by data loader
                    {'baseline': torch.Tensor         Has to be converted back to datetime
                    'y': torch.Tensor
                    'x': torch.Tensor}
        """
        s_num_lead_time_steps = self.settings['s_num_lead_time_steps']
        s_target_height_width = self.settings['s_target_height_width']

        # Unpacking outputs -> except for loss they are all batched tensors
        target_normed = outputs['target']
        pred_model_one_hot = outputs['pred']
        baseline = outputs['baseline']

        # Model Prediction

        # pred_model_softmaxed = torch.nn.Softmax(dim=1)(pred_model_one_hot)
        # pred_model_argmaxed = torch.argmax(pred_model_softmaxed, dim=1)

        # TODO: AM I DOING THIS FOR THE PREDICTION PIPELINE TOO?
        # Converting prediction from one-hot to (lognormed) mm
        _, _, linspace_binning = pl_module._linspace_binning_params
        pred_model_normed = one_hot_to_lognormed_mm(pred_model_one_hot, linspace_binning, channel_dim=1)

        # Inverse normalize target and prediction

        pred_model_mm = inverse_normalize_data(pred_model_normed,
                                         pl_module.mean_filtered_log_data,
                                         pl_module.std_filtered_log_data)

        target_mm = inverse_normalize_data(target_normed,
                                        pl_module.mean_filtered_log_data,
                                        pl_module.std_filtered_log_data)

        pred_baseline_mm = baseline[:, s_num_lead_time_steps, :, :]
        pred_baseline_mm = T.CenterCrop(size=s_target_height_width)(pred_baseline_mm)

        # Double-checked alignment visually (See apple notes Science/testing code/Testing on predict_batch_end())

        self.evaluate(pred_baseline_mm, pred_model_mm, target_mm, pl_module)


    def evaluate(
            self,
            pred_model_mm,
            pred_baseline_mm,
            target,
            pl_module,
    ):
        """
        Input:
            pred: torch.Tensor
                shape: [batch, height, width]
            target: torch.Tensor
                shape: [batch, height, width]
            pl_module: pl.LightningModule
        """







def ckpt_quick_eval_with_baseline(
        model,
        checkpoint_name,
        sample_coords,
        radolan_statistics_dict,
        linspace_binning_params,

        ckpt_settings,  # Make sure to pass the settings of the checkpoint
        s_batch_size,
        s_num_workers_data_loader,
        **__,
):
    """
    Input:
        model:
            Model to evaluate, which has been loaded from checkpoint
        checkpoint_name: str
            The name of the checkpoint from which the model has been loded ... used later for naming the saved files
        sample_coords: np.array: Coordinate space

            - generated by patch_indecies_to_sample_coords() -
            array of arrays with valid patch coordinates

            shape: [num_valid_patches, num_dims=3]
            [
            [np.datetime64 target frame,
            slice of y coordinates,
            slice of x coordinates],
            ...]
    """
    # Setting model to baseline mode, which chooses thr right predict_step() method
    model.set_mode(mode='baseline')

    #  Data Set
    data_set_eval_filtered = FilteredDatasetXr(
        sample_coords,
        radolan_statistics_dict,
        mode='baseline',
        settings=ckpt_settings,
        data_into_ram=False,
        baseline_path=ckpt_settings['s_baseline_path'],
        baseline_variable_name=ckpt_settings['s_baseline_variable_name'],
        num_input_frames_baseline=ckpt_settings['s_num_input_frames_baseline'],
    )

    # Boolean stating whether samples have input padding:
    # If they do have padding, this is going to be removed by center cropping
    samples_have_padding = data_set_eval_filtered.samples_have_padding

    # Data Loader
    data_loader_eval_filtered = DataLoader(
        data_set_eval_filtered,
        shuffle=False,
        batch_size=s_batch_size,
        drop_last=True,
        num_workers=s_num_workers_data_loader,
        pin_memory=True,
    )

    # Callbacks
    evaluate_baseline_callback = EvaluateBaselineCallback(
            linspace_binning_params,
            checkpoint_name,
            samples_have_padding,
            ckpt_settings,
    )

    trainer = pl.Trainer(
        callbacks=evaluate_baseline_callback,
    )

    trainer.predict(
        model=model,
        dataloaders=data_loader_eval_filtered,
        return_predictions=False  # By default, lightning aggregates the output of all batches, disable this to prevent memory overflow
    )







