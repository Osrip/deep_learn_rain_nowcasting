import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from helper.pre_process_target_input import one_hot_to_lognormed_mm, inverse_normalize_data
from helper.helper_functions import move_to_device
from helper.memory_logging import print_gpu_memory, print_ram_usage, format_duration
import time
import pandas as pd
import os


class EvaluateBaselineCallback(pl.Callback):

    def __init__(
            self,
            linspace_binning_params,
            checkpoint_name,
            dataset_name,
            samples_have_padding,
            settings,
    ):
        '''
        This callback calculates the evaluation metrics
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
        self.dataset_name = dataset_name
        self.samples_have_padding = samples_have_padding

        # Add sample counter to track samples across batches
        self.sample_idx_counter = 0

        # Initialising logging lists
        self.sample_indices = []  # Add this to track sample indices
        self.losses_model = []
        self.rmses_model = []
        self.rmses_baseline = []
        self.l1_differences_model = []
        self.l1_differences_baseline = []
        self.means_target = []
        self.means_pred_model = []
        self.means_pred_baseline = []
        self.certainties_target_bin_model = []
        self.certainties_max_pred = []
        self.stds_binning_model = []

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
        if batch_idx <= 2:
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

        if not isinstance(pl_module.loss_func, torch.nn.CrossEntropyLoss):
            raise ValueError('Loss was expected to be xrossentropy in Validation')

        model_losses = torch.nn.CrossEntropyLoss(reduction='none')(pred_model_binned_no_smax,
                                                                   torch.argmax(target_one_hot, dim=1))  # [batch]
        model_losses = model_losses.mean(dim=(1, 2))

        rmse_model_per_sample = torch.sqrt(torch.mean((pred_model_mm - target_mm) ** 2, dim=(1, 2)))  # [batch]
        rmse_baseline_per_sample = torch.sqrt(torch.nanmean((pred_baseline_mm - target_mm) ** 2, dim=(1, 2)))  # [batch]

        l1_difference_model_per_sample = torch.abs(pred_model_mm - target_mm).mean(dim=(1, 2))
        l1_difference_baseline_per_sample = torch.abs(pred_baseline_mm - target_mm).nanmean(dim=(1, 2))

        mean_target_per_sample = target_mm.mean(dim=(1, 2))  # [batch]
        mean_pred_model_per_sample = pred_model_mm.mean(dim=(1, 2))  # [batch]
        mean_pred_baseline_per_sample = pred_baseline_mm.nanmean(dim=(1, 2))  # [batch]

        # Certainty per sample
        certainty_target_bin_per_sample = pred_probs_correct.mean(dim=(1, 2, 3))  # [batch]
        certainty_max_pred = pred_model_binned_smax.max(dim=1).values.mean(dim=(1, 2))

        # Std per sample
        std_binning_model_per_sample = pred_model_binned_no_smax.std(dim=1)
        std_binning_model_per_sample = std_binning_model_per_sample.mean(dim=(1, 2))  # [batch]

        # Get batch size to increment sample indices
        batch_size = pred_model_mm.shape[0]

        # Create sample indices for this batch
        batch_sample_indices = list(range(self.sample_idx_counter, self.sample_idx_counter + batch_size))
        self.sample_idx_counter += batch_size

        # Append sample indices and metrics
        self.sample_indices.extend(batch_sample_indices)
        self.losses_model.extend(model_losses.tolist())
        self.rmses_model.extend(rmse_model_per_sample.tolist())
        self.rmses_baseline.extend(rmse_baseline_per_sample.tolist())
        self.l1_differences_model.extend(l1_difference_model_per_sample.tolist())
        self.l1_differences_baseline.extend(l1_difference_baseline_per_sample.tolist())
        self.means_target.extend(mean_target_per_sample.tolist())
        self.means_pred_model.extend(mean_pred_model_per_sample.tolist())
        self.means_pred_baseline.extend(mean_pred_baseline_per_sample.tolist())
        self.certainties_max_pred.extend(certainty_max_pred.tolist())
        self.certainties_target_bin_model.extend(certainty_target_bin_per_sample.tolist())
        self.stds_binning_model.extend(std_binning_model_per_sample.tolist())

    def save_evaluations_logs(self):
        s_dirs = self.settings['s_dirs']
        log_dir = s_dirs['logs']
        evaluation_dir = os.path.join(log_dir, "evaluation")

        # Check if the evaluation directory exists, create it if not
        os.makedirs(evaluation_dir, exist_ok=True)

        # Remove ".ckpt" from checkpoint_name if present
        checkpoint_name_cleaned = self.checkpoint_name.replace(".ckpt", "")

        # Convert metrics to a DataFrame - now include sample_idx
        df = pd.DataFrame({
            "sample_idx": self.sample_indices,  # Add sample index
            "losses_model": self.losses_model,
            "rmses_model": self.rmses_model,
            "rmses_baseline": self.rmses_baseline,
            "l1_difference_model": self.l1_differences_model,
            "l1_difference_baseline": self.l1_differences_baseline,
            "means_target": self.means_target,
            "means_pred_model": self.means_pred_model,
            "means_pred_baseline": self.means_pred_baseline,
            "certainties_target_bin_model": self.certainties_target_bin_model,
            "certainties_max_pred": self.certainties_max_pred,
            "stds_binning_model": self.stds_binning_model,
        })

        # Define CSV file path based on checkpoint_name
        csv_file = os.path.join(evaluation_dir,
                                f"dataset_{self.dataset_name}_ckpt_{checkpoint_name_cleaned}_metrics.csv")

        # Save DataFrame to CSV
        df.to_csv(csv_file, index=False)
        print(f"Saved evaluation logs to {csv_file}")

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.save_evaluations_logs()

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
        # Remove ".ckpt" from checkpoint_name if present
        checkpoint_name_cleaned = self.checkpoint_name.replace(".ckpt", "")

        step_start_time = time.time()
        s_dirs = self.settings['s_dirs']
        batches_outputs_dir = s_dirs['batches_outputs']
        batches_outputs_subdir = f'dataset_{self.dataset_name}_ckpt_{checkpoint_name_cleaned}'
        save_name_batches = f'batch_{batch_idx:04d}.pt'
        save_name_outputs = f'outputs_{batch_idx:04d}.pt'

        # Create the subdirectory (and parents if needed)
        subdir_full_path = os.path.join(batches_outputs_dir, batches_outputs_subdir)
        os.makedirs(subdir_full_path, exist_ok=True)

        save_path_batches = os.path.join(subdir_full_path, save_name_batches)
        save_path_outputs = os.path.join(subdir_full_path, save_name_outputs)

        # Save batch
        keys = ['dynamic', 'static', 'baseline']
        batch_dict = {key: en for key, en in zip(keys, batch)}
        batch_dict = move_to_device(batch_dict, device='cpu')
        torch.save(batch_dict, save_path_batches)

        # Save outputs
        outputs = move_to_device(outputs, device='cpu')
        torch.save(outputs, save_path_outputs)
        print(f'\n Done saving the batch. Took {format_duration(time.time() - step_start_time)} \n')