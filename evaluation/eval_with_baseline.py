import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from helper.pre_process_target_input import one_hot_to_lognormed_mm, inverse_normalize_data, img_one_hot, normalize_data
from helper.helper_functions import move_to_device
from helper.memory_logging import print_gpu_memory, print_ram_usage, format_duration
from helper.dlbd import dlbd_target_pre_processing
import time
import pandas as pd
import os


class EvaluateBaselineCallback(pl.Callback):
    class EvaluateBaselineCallback(pl.Callback):

        def __init__(
                self,
                linspace_binning_params,
                checkpoint_name,
                dataset_name,
                samples_have_padding,
                settings,
                sigmas_dlbd_eval=None,  # Renamed parameter
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
                sigmas_dlbd_eval: list of float, optional
                    List of sigma values for DLBD evaluation. If None, DLBD evaluation is skipped.
            '''

            super().__init__()
            self.settings = settings
            self.linspace_binning_params = linspace_binning_params
            self.checkpoint_name = checkpoint_name
            self.dataset_name = dataset_name
            self.samples_have_padding = samples_have_padding

            # Initialize sigmas for DLBD evaluation
            if sigmas_dlbd_eval is None:
                self.sigmas_dlbd_eval = []
                self.do_dlbd_eval = False
            else:
                self.sigmas_dlbd_eval = list(sigmas_dlbd_eval)
                self.do_dlbd_eval = True

            # Check if training sigma should be included in evaluation
            training_sigma = self.settings.get('s_sigma_target_smoothing', None)
            if self.do_dlbd_eval and training_sigma is not None and training_sigma not in self.sigmas_dlbd_eval:
                self.sigmas_dlbd_eval.append(training_sigma)

            # Sort the sigmas for consistency
            self.sigmas_dlbd_eval = sorted(self.sigmas_dlbd_eval)

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

            # Initialize lists for DLBD metrics with separate model and baseline tracking
            self.dlbd_metrics = {}
            for sigma in self.sigmas_dlbd_eval:
                sigma_str = f"{sigma:.2f}"
                self.dlbd_metrics[f"dlbd_model_sigma_{sigma_str}"] = []
                self.dlbd_metrics[f"dlbd_baseline_sigma_{sigma_str}"] = []

            # Define precipitation thresholds for categorical metrics
            self.precip_thresholds = [0.5, 2.0, 8.0, 20.0]  # in mm/h

            # Initialize dictionaries to store CSI and Accuracy metrics
            self.csi_model = {thr: [] for thr in self.precip_thresholds}
            self.csi_baseline = {thr: [] for thr in self.precip_thresholds}
            self.acc_model = {thr: [] for thr in self.precip_thresholds}
            self.acc_baseline = {thr: [] for thr in self.precip_thresholds}

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
                'baseline'
                'target_uncropped' (for DLBD eval)
                'target_normalized_uncropped' (for DLBD eval)
                }
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

        # Get uncropped versions for DLBD eval if available
        target_normed_uncropped = outputs.get('target_normalized_uncropped', None)

        # Model Prediction
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

        # Center crop baseline for regular metrics
        pred_baseline_mm = baseline
        pred_baseline_mm_cropped = T.CenterCrop(size=s_target_height_width)(pred_baseline_mm)

        # Check if we have uncropped targets for DLBD evaluation
        if self.do_dlbd_eval:
            if target_normed_uncropped is None:
                # If we don't have uncropped targets, we can't do DLBD eval
                print(
                    "Warning: DLBD evaluation is enabled but no uncropped targets provided. DLBD evaluation will be skipped.")
                self.do_dlbd_eval = False

        # Evaluate with standard metrics (using cropped tensors)
        self.evaluate_batch(
            pred_model_mm,
            pred_model_binned_no_smax,
            pred_baseline_mm_cropped,
            target_mm,
            target_one_hot,
            loss,
            pl_module,
            target_normed_uncropped
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
            target_normed_uncropped=None,
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
            target_normed_uncropped: torch.Tensor, optional
                Uncropped normalized target for DLBD evaluation
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

        # Calculate CSI and Accuracy metrics for each threshold and each sample
        for thr in self.precip_thresholds:
            batch_size = pred_model_mm.shape[0]

            # Iterate over each sample in the batch
            for b in range(batch_size):
                # Convert tensors to numpy arrays for pysteps functions
                pred_model_np = pred_model_mm[b].detach().cpu().numpy()
                pred_baseline_np = pred_baseline_mm[b].detach().cpu().numpy()
                target_np = target_mm[b].detach().cpu().numpy()

                # Handle NaNs in the data
                pred_model_np = np.nan_to_num(pred_model_np, nan=0.0)
                pred_baseline_np = np.nan_to_num(pred_baseline_np, nan=0.0)
                target_np = np.nan_to_num(target_np, nan=0.0)

                # Calculate categorical scores for model
                model_scores = detcatscores.det_cat_fct(pred_model_np, target_np, thr, scores=["CSI", "ACC"])
                self.csi_model[thr].append(model_scores["CSI"])
                self.acc_model[thr].append(model_scores["ACC"])

                # Calculate categorical scores for baseline
                baseline_scores = detcatscores.det_cat_fct(pred_baseline_np, target_np, thr, scores=["CSI", "ACC"])
                self.csi_baseline[thr].append(baseline_scores["CSI"])
                self.acc_baseline[thr].append(baseline_scores["ACC"])

        # Calculate DLBD metrics for each sigma value if enabled
        if self.do_dlbd_eval and target_normed_uncropped is not None:
            # [DLBD evaluation code remains unchanged]
            s_target_height_width = self.settings['s_target_height_width']

            # Create one-hot encoded uncropped target using the same binning
            _, _, linspace_binning = self.linspace_binning_params
            s_num_bins_crossentropy = self.settings['s_num_bins_crossentropy']

            # Create one-hot encoded target from the uncropped normalized target
            target_one_hot_uncropped = img_one_hot(target_normed_uncropped, s_num_bins_crossentropy, linspace_binning)
            target_one_hot_uncropped = target_one_hot_uncropped.float()  # Convert to float
            target_one_hot_uncropped = torch.permute(target_one_hot_uncropped, (0, 3, 1, 2))  # b w h c -> b c w h

            # We need to convert baseline mm values to one-hot for DLBD calculation
            # First normalize baseline
            baseline_normed = normalize_data(
                pred_baseline_mm,
                pl_module.mean_filtered_log_data,
                pl_module.std_filtered_log_data
            )

            # Convert to one-hot
            baseline_one_hot = img_one_hot(baseline_normed, s_num_bins_crossentropy, linspace_binning)
            baseline_one_hot = baseline_one_hot.float()  # Convert to float
            baseline_one_hot = torch.permute(baseline_one_hot, (0, 3, 1, 2))  # b w h c -> b c w h

            # Convert to logits-like format for cross-entropy calculation
            # Explicitly create as float32 tensor
            baseline_logits = torch.zeros_like(baseline_one_hot, dtype=torch.float32).to(baseline_one_hot.device)
            max_indices = torch.argmax(baseline_one_hot, dim=1, keepdim=True)
            baseline_logits.scatter_(1, max_indices, 10.0)  # Large value at the max index

            for sigma in self.sigmas_dlbd_eval:
                # [DLBD calculations remain unchanged]
                sigma_str = f"{sigma:.2f}"

                # Apply gaussian smoothing to the uncropped one-hot target for current sigma
                target_one_hot_blurred = dlbd_target_pre_processing(
                    input_tensor=target_one_hot_uncropped,
                    output_size=s_target_height_width,
                    sigma=sigma,
                    kernel_size=None
                )

                # 1. Calculate DLBD for model predictions
                dlbd_losses_model = torch.nn.CrossEntropyLoss(reduction='none')(
                    pred_model_binned_no_smax,
                    torch.argmax(target_one_hot_blurred, dim=1)
                )
                dlbd_losses_model_per_sample = dlbd_losses_model.mean(dim=(1, 2))

                # 2. Calculate DLBD for baseline predictions
                dlbd_losses_baseline = torch.nn.CrossEntropyLoss(reduction='none')(
                    baseline_logits,
                    torch.argmax(target_one_hot_blurred, dim=1)
                )
                dlbd_losses_baseline_per_sample = dlbd_losses_baseline.mean(dim=(1, 2))

                # Store metrics with clear naming
                self.dlbd_metrics[f"dlbd_model_sigma_{sigma_str}"].extend(dlbd_losses_model_per_sample.tolist())
                self.dlbd_metrics[f"dlbd_baseline_sigma_{sigma_str}"].extend(dlbd_losses_baseline_per_sample.tolist())

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

        # Create a dictionary with all metrics
        metrics_dict = {
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
        }

        # Add CSI and Accuracy metrics to the dictionary
        for thr in self.precip_thresholds:
            metrics_dict[f"csi_model_{thr:.1f}mm"] = self.csi_model[thr]
            metrics_dict[f"csi_baseline_{thr:.1f}mm"] = self.csi_baseline[thr]
            metrics_dict[f"acc_model_{thr:.1f}mm"] = self.acc_model[thr]
            metrics_dict[f"acc_baseline_{thr:.1f}mm"] = self.acc_baseline[thr]

        # Add DLBD metrics to the dictionary
        if self.do_dlbd_eval:
            for metric_key, dlbd_values in self.dlbd_metrics.items():
                if dlbd_values:  # Only add non-empty lists
                    metrics_dict[metric_key] = dlbd_values

        # Convert metrics to a DataFrame
        df = pd.DataFrame(metrics_dict)

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