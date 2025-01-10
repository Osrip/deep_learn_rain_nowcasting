import torch
from torch.utils.data import DataLoader
from load_data_xarray import FilteredDatasetXr
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import Subset
from helper.pre_process_target_input import one_hot_to_lognormed_mm, inverse_normalize_data
from helper.helper_functions import move_to_device
from helper.memory_logging import print_gpu_memory, print_ram_usage, format_duration
import time
import random
import pandas as pd
import os


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
            ...
            samples_have_padding: bool
                If True this indicates an input padding, therefore we will center crop to s_width_height
        '''

        super().__init__()
        self.settings = settings
        self.linspace_binning_params = linspace_binning_params
        self.checkpoint_name = checkpoint_name
        self.samples_have_padding = samples_have_padding

        # Initialising logging lists
        self.losses_model = []
        # self.losses_baseline = []

        self.rmses_model = []
        self.rmses_baseline = []

        self.means_target = []
        self.means_pred_model = []
        self.means_pred_baseline = []

        self.certainties_target_bin_model = []
        self.certainties_max_pred = []
        self.stds_model = []

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
                'target:
                'target_binned'
                'baseline'}

        """

        print_gpu_memory()
        print_ram_usage()

        # Save certain batches for plotting:
        # TODO: FIXING FREEZING
        if batch_idx <= 4:
            self.save_batch_output(batch, outputs, batch_idx)

        s_num_lead_time_steps = self.settings['s_num_lead_time_steps']
        s_target_height_width = self.settings['s_target_height_width']

        # Unpacking outputs -> except for loss they are all batched tensors
        loss = outputs['loss']
        pred_model_binned_no_smax = outputs['pred']
        target_normed = outputs['target']
        target_one_hot = outputs['target_binned']
        baseline = outputs['baseline']

        # Model Prediction

        # pred_model_softmaxed = torch.nn.Softmax(dim=1)(pred_model_binned_no_smax)
        # pred_model_argmaxed = torch.argmax(pred_model_softmaxed, dim=1)

        # TODO: AM I DOING THIS FOR THE PREDICTION PIPELINE TOO?
        # Converting prediction from one-hot to (lognormed) mm
        _, _, linspace_binning = pl_module._linspace_binning_params
        pred_model_normed = one_hot_to_lognormed_mm(pred_model_binned_no_smax, linspace_binning, channel_dim=1)

        # Inverse normalize target and prediction

        pred_model_mm = inverse_normalize_data(pred_model_normed,
                                         pl_module.mean_filtered_log_data,
                                         pl_module.std_filtered_log_data)

        target_mm = inverse_normalize_data(target_normed,
                                        pl_module.mean_filtered_log_data,
                                        pl_module.std_filtered_log_data)

        pred_baseline_mm = baseline
        # pred_baseline_mm = baseline[:, s_num_lead_time_steps, :, :]

        pred_baseline_mm = T.CenterCrop(size=s_target_height_width)(pred_baseline_mm)

        # Double-checked alignment visually (See apple notes Science/testing code/Testing on predict_batch_end())

        self.evaluate_batch(
            pred_model_mm,
            pred_model_binned_no_smax,
            pred_baseline_mm,
            target_mm,
            target_one_hot,
            loss,
            pl_module,
        )


    def evaluate_batch(
            self,
            pred_model_mm,
            pred_model_binned_no_smax,
            pred_baseline_mm,
            target_mm,
            target_one_hot,
            loss,
            pl_module,
    ):
        """
        Input:
            pred_model_mm: torch.Tensor
                shape: [batch, height, width]
            pred_model_binned: torch.Tensor
                shape: [batch, channels, height, width]
            pred_baseline_mm: torch.Tensor
                shape: [batch, height, width]
            target_mm: torch.Tensor
                shape: [batch, height, width]
            target_one_hot: torch.Tensor
                shape: [batch, channels, height, width]
            loss: torch.Tensor (single scalar) representing batch loss from the model
        """

        # Check if pred_model_binned is already soft maxed
        sum_along_channels = pred_model_binned_no_smax.sum(dim=1, keepdim=False)
        if torch.allclose(sum_along_channels, torch.ones_like(sum_along_channels), atol=1e-3):
            raise ValueError('Looks like the prediction has been softmaxed already, expected non-softmaxed predictions')
        else:
            pred_model_binned_smax = torch.softmax(pred_model_binned_no_smax, dim=1)

        # Calculations for certainty of the target bin (Probability at the bin that would have bin correct)
        # Find the ground truth bin index
        bin_idx = target_one_hot.argmax(dim=1, keepdim=True)  # shape: [batch, 1, height, width]
        # Gather probabilities of the correct bin in the target
        pred_probs_correct = pred_model_binned_smax.gather(dim=1, index=bin_idx)  # shape: [batch, 1, height, width]


        # TODO: Make this XEntropy loss, ideally drawing loss function directly from pl_modul
        #  We have to do this again, as the loss is only calculated for the whole batch.
        if not isinstance(pl_module.loss_func, torch.nn.CrossEntropyLoss):
            raise ValueError('Loss was expected to be xrossentropy in Validation')

        model_losses = torch.nn.CrossEntropyLoss(reduction='none')(pred_model_binned_no_smax, torch.argmax(target_one_hot, dim=1))  # [batch]
        model_losses = model_losses.mean(dim=(1,2))
        # no XEntropy loss on deterministic baseline

        rmse_model_per_sample = torch.sqrt(torch.mean((pred_model_mm - target_mm) ** 2, dim=(1,2)))  # [batch]
        rmse_baseline_per_sample = torch.sqrt(torch.mean((pred_baseline_mm - target_mm) ** 2, dim=(1,2)))  # [batch]

        mean_target_per_sample = target_mm.mean(dim=(1,2))  # [batch]
        mean_pred_model_per_sample = pred_model_mm.mean(dim=(1,2))  # [batch]
        mean_pred_baseline_per_sample = pred_baseline_mm.mean(dim=(1,2))  # [batch]

        # Certainty per sample
        # Probability for correct bin in target (not good measure)
        certainty_target_bin_per_sample = pred_probs_correct.mean(dim=(1,2,3))  # [batch]
        # Probability
        certainty_max_pred = pred_model_binned_smax.max(dim=1).values.mean(dim=(1,2))

        # Std per sample (std across channels, then average spatial dims)
        # First: std across channels -> shape: [batch, height, width]
        std_model_per_sample = pred_model_binned_no_smax.std(dim=1)
        # Average spatially to get a single scalar per sample
        std_model_per_sample = std_model_per_sample.mean(dim=(1,2))  # [batch]

        # Append per-sample metrics to the logging lists
        self.losses_model.extend(model_losses.tolist())
        # self.losses_baseline.extend(baseline_losses.tolist())

        self.rmses_model.extend(rmse_model_per_sample.tolist())
        self.rmses_baseline.extend(rmse_baseline_per_sample.tolist())

        self.means_target.extend(mean_target_per_sample.tolist())
        self.means_pred_model.extend(mean_pred_model_per_sample.tolist())
        self.means_pred_baseline.extend(mean_pred_baseline_per_sample.tolist())

        self.certainties_max_pred.extend(certainty_max_pred.tolist())
        self.certainties_target_bin_model.extend(certainty_target_bin_per_sample.tolist())
        self.stds_model.extend(std_model_per_sample.tolist())

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.save_evaluations_logs()

    def save_evaluations_logs(self):

        s_dirs = self.settings['s_dirs']
        log_dir = s_dirs['logs']
        evaluation_dir = os.path.join(log_dir, "evaluation")

        # Check if the evaluation directory exists, create it if not
        os.makedirs(evaluation_dir, exist_ok=True)

        # Remove ".ckpt" from checkpoint_name if present
        checkpoint_name_cleaned = self.checkpoint_name.replace(".ckpt", "")

        # Convert metrics to a DataFrame
        df = pd.DataFrame({
            "losses_model":         self.losses_model,
            # "losses_baseline":      self.losses_baseline,
            "rmses_model":          self.rmses_model,
            "rmses_baseline":       self.rmses_baseline,
            "means_target":         self.means_target,
            "means_pred_model":     self.means_pred_model,
            "means_pred_baseline":  self.means_pred_baseline,
            "certainties_target_bin_model":    self.certainties_target_bin_model,
            "stds_model":           self.stds_model,
        })

        # Define CSV file path based on checkpoint_name
        csv_file = os.path.join(evaluation_dir, f"{checkpoint_name_cleaned}_metrics.csv")

        # Save DataFrame to CSV
        df.to_csv(csv_file, index=False)
        print(f"Saved evaluation logs to {csv_file}")


    def save_batch_output(self, batch, outputs, batch_idx):
        '''
        Input
            batch: list of dicts of tensors
                The list contains the variable_dicts:
                    [dynamic_variable_dict, static_variable_dict, baseline_dict]
            outputs: dict of tensors
                The dict contains the tensors:
                    {'loss': torch.Tensor,
                    'pred': torch.Tensor,
                    'target: torch.Tensor,
                    'target_binned' torch.Tensor,
                    'baseline': torch.Tensor}
        '''
        print(f"\n Saving batch number {batch_idx} \n ...")
        step_start_time = time.time()
        s_dirs = self.settings['s_dirs']
        batches_outputs_dir = s_dirs['batches_outputs']
        save_name_batches = f'batch_{batch_idx:04d}.pt'
        save_name_outputs = f'outputs_{batch_idx:04d}.pt'
        save_path_batches = os.path.join(batches_outputs_dir, save_name_batches)
        save_path_outputs = os.path.join(batches_outputs_dir, save_name_outputs)

        # Save batch
        keys = ['dynamic', 'static', 'baseline']
        batch_dict = {key: en for key, en in zip(keys, batch)}
        batch_dict = move_to_device(batch_dict, device='cpu')
        torch.save(batch_dict, save_path_batches)

        # Save outputs
        outputs = move_to_device(outputs, device='cpu')
        torch.save(outputs, save_path_outputs)
        print(f'\n Done saving the batch. Took {format_duration(time.time() - step_start_time)} \n')


def ckpt_quick_eval_with_baseline(
        model,
        checkpoint_name,
        sample_coords,
        radolan_statistics_dict,
        linspace_binning_params,

        ckpt_settings,  # Make sure to pass the settings of the checkpoint
        s_batch_size,
        s_baseline_path,
        s_num_workers_data_loader,

        subsample_dataset_to_len=1280, #=1280, #1280, #=50,

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
    print(f'Baseline path is {s_baseline_path}')
    print('Set model mode')
    # Setting model to baseline mode, which chooses the right predict_step() method
    model.set_mode(mode='baseline')

    print('Initialize Dataset')

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

    # Subsampling
    sub_sampled = False
    if subsample_dataset_to_len is not None:
        if subsample_dataset_to_len < len(data_set_eval_filtered):
            print(f'Randomly subsample Dataset from length {len(data_set_eval_filtered)} to len {subsample_dataset_to_len}')
            # Randomly subsample dataset
            subset_indices = random.sample(range(len(data_set_eval_filtered)), subsample_dataset_to_len)
            # subset_indices = list(range(crop_dataset_to_len))  # Choose the first `desired_sample_size` samples
            data_set_eval_filtered = Subset(data_set_eval_filtered, subset_indices).dataset
            sub_sampled = True

    if not sub_sampled:
        print(f'Len of dataset is {subsample_dataset_to_len}')


    print('Load "samples_have_padding"')
    # Boolean stating whether samples have input padding:F
    # If they do have padding, this is going to be removed by center cropping
    samples_have_padding = data_set_eval_filtered.samples_have_padding

    print('Initializing Dataloader')

    # Data Loader
    # THIS FIXES FREEZING ISSUE!
    data_loader_eval_filtered = DataLoader(
        data_set_eval_filtered,
        shuffle=True,
        batch_size=s_batch_size,
        drop_last=True,
        num_workers=0, # EITHER THIS
        pin_memory=False, # OR THIS FIXES FREEZING
        # timeout=0,  # TODO: Potentially try this to see whether the freezing happens during batch loading
    )

    # Original Data Loader ---> THIS CAUSES GETTING STUCK
    # data_loader_eval_filtered = DataLoader(
    #     data_set_eval_filtered,
    #     shuffle=False,
    #     batch_size=s_batch_size,
    #     drop_last=True,
    #     num_workers=s_num_workers_data_loader,
    #     pin_memory=True,
    # )

    print('Initialising Callback')

    # Callbacks
    evaluate_baseline_callback = EvaluateBaselineCallback(
            linspace_binning_params,
            checkpoint_name,
            samples_have_padding,
            ckpt_settings,
    )

    print('Initializing Trainer')

    trainer = pl.Trainer(
        callbacks=evaluate_baseline_callback,
    )

    print('Starting evaluation with trainer.predict')

    trainer.predict(
        model=model,
        dataloaders=data_loader_eval_filtered,
        return_predictions=False  # By default, lightning aggregates the output of all batches, disable this to prevent memory overflow
    )







