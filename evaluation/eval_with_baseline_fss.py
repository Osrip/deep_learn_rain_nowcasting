import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import os
from pysteps import verification
from helper.pre_process_target_input import one_hot_to_lognormed_mm, inverse_normalize_data
from helper.memory_logging import print_gpu_memory, print_ram_usage
import torchvision.transforms as T


class FSSEvaluationCallback(pl.Callback):
    """
    Callback for calculating Fractions Skill Score (FSS) across different scales and thresholds.
    FSS is calculated for both model predictions and baseline predictions against the ground truth.

    This callback focuses exclusively on FSS calculation and does not duplicate batch saving
    functionality provided by EvaluateBaselineCallback.
    """

    def __init__(
            self,
            scales,
            thresholds,
            linspace_binning_params,
            checkpoint_name,
            dataset_name,
            settings,
    ):
        '''
        Args:
            scales (list): List of spatial scales (neighborhood sizes) for FSS calculation
            thresholds (list): List of precipitation thresholds for FSS calculation in mm/h
            linspace_binning_params: Parameters for converting model output to precipitation values
            checkpoint_name (str): Name of the checkpoint being evaluated
            dataset_name (str): Name of the dataset being evaluated
            settings (dict): Settings dictionary
        '''
        super().__init__()
        self.scales = scales
        self.thresholds = thresholds
        self.linspace_binning_params = linspace_binning_params
        self.checkpoint_name = checkpoint_name
        self.dataset_name = dataset_name
        self.settings = settings

        # Get the FSS calculation method from pysteps
        self.fss_calc = verification.get_method("FSS")

        # Initialize dictionaries to store FSS values
        # Structure: {threshold: {scale: [fss_values_for_each_batch]}}
        self.fss_model = {threshold: {scale: [] for scale in scales} for threshold in thresholds}
        self.fss_baseline = {threshold: {scale: [] for scale in scales} for threshold in thresholds}

    def on_predict_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int = 0
    ):
        """
        Process batch results and calculate FSS for different scales and thresholds.
        """
        print_gpu_memory()
        print_ram_usage()

        s_target_height_width = self.settings['s_target_height_width']

        # Unpacking outputs
        pred_model_binned_no_smax = outputs['pred']
        target_normed = outputs['target']
        baseline = outputs['baseline']

        # Converting model prediction from binned to mm values
        _, _, linspace_binning = pl_module._linspace_binning_params
        pred_model_normed = one_hot_to_lognormed_mm(pred_model_binned_no_smax, linspace_binning, channel_dim=1)

        # Inverse normalize to get actual precipitation values in mm/h
        pred_model_mm = inverse_normalize_data(
            pred_model_normed,
            pl_module.mean_filtered_log_data,
            pl_module.std_filtered_log_data
        )

        target_mm = inverse_normalize_data(
            target_normed,
            pl_module.mean_filtered_log_data,
            pl_module.std_filtered_log_data
        )

        # Prepare baseline prediction
        pred_baseline_mm = baseline
        pred_baseline_mm = T.CenterCrop(size=s_target_height_width)(pred_baseline_mm)

        # Calculate FSS for this batch
        self.evaluate_batch_fss(pred_model_mm, pred_baseline_mm, target_mm)

    def evaluate_batch_fss(self, pred_model_mm, pred_baseline_mm, target_mm):
        """
        Calculate FSS for a batch across all scales and thresholds.

        Args:
            pred_model_mm (torch.Tensor): Model predictions in mm/h
            pred_baseline_mm (torch.Tensor): Baseline predictions in mm/h
            target_mm (torch.Tensor): Target values in mm/h
        """
        # Convert tensors to numpy for pysteps
        pred_model_np = pred_model_mm.detach().cpu().numpy()
        pred_baseline_np = pred_baseline_mm.detach().cpu().numpy()
        target_np = target_mm.detach().cpu().numpy()

        # Process each sample in the batch
        batch_size = pred_model_np.shape[0]

        # For each sample in batch
        for sample_idx in range(batch_size):
            forecast_model = pred_model_np[sample_idx]
            forecast_baseline = pred_baseline_np[sample_idx]
            observation = target_np[sample_idx]

            # For each threshold and scale combination
            for threshold in self.thresholds:
                for scale in self.scales:
                    # Calculate FSS for model prediction
                    try:
                        # Handle potential edge cases
                        if np.all(forecast_model <= threshold) and np.all(observation <= threshold):
                            # When both fields have no precipitation above threshold
                            fss_model = 1.0  # Perfect score
                        else:
                            # Use pysteps FSS calculation
                            fss_model = self.fss_calc(forecast_model, observation, threshold, scale)
                    except Exception as e:
                        print(f"Error calculating model FSS for threshold {threshold}, scale {scale}: {e}")
                        fss_model = np.nan

                    # Calculate FSS for baseline prediction
                    try:
                        if np.all(forecast_baseline <= threshold) and np.all(observation <= threshold):
                            # When both fields have no precipitation above threshold
                            fss_baseline = 1.0  # Perfect score
                        else:
                            fss_baseline = self.fss_calc(forecast_baseline, observation, threshold, scale)
                    except Exception as e:
                        print(f"Error calculating baseline FSS for threshold {threshold}, scale {scale}: {e}")
                        fss_baseline = np.nan

                    # Store FSS values
                    self.fss_model[threshold][scale].append(fss_model)
                    self.fss_baseline[threshold][scale].append(fss_baseline)

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """
        Save FSS evaluations at the end of the prediction epoch.
        """
        self.save_fss_evaluations()

    def save_fss_evaluations(self):
        """
        Save FSS evaluations to CSV files.
        Creates two files: one for model FSS and one for baseline FSS.
        Each CSV has thresholds as rows and scales as columns.
        """
        s_dirs = self.settings['s_dirs']
        log_dir = s_dirs['logs']
        evaluation_dir = os.path.join(log_dir, "evaluation")
        fss_dir = os.path.join(evaluation_dir, "fss")

        # Create evaluation directory if it doesn't exist
        os.makedirs(fss_dir, exist_ok=True)

        # Remove ".ckpt" from checkpoint_name if present
        checkpoint_name_cleaned = self.checkpoint_name.replace(".ckpt", "")

        # Create DataFrames for model and baseline FSS
        # Format: Rows = thresholds, Columns = scales
        df_model = pd.DataFrame(index=self.thresholds, columns=[f"scale_{scale}" for scale in self.scales])
        df_baseline = pd.DataFrame(index=self.thresholds, columns=[f"scale_{scale}" for scale in self.scales])

        # Also create DataFrames for FSS improvement (model vs baseline)
        df_improvement = pd.DataFrame(index=self.thresholds, columns=[f"scale_{scale}" for scale in self.scales])

        # Fill DataFrames with average FSS values
        for threshold in self.thresholds:
            for scale in self.scales:
                model_fss_values = self.fss_model[threshold][scale]
                baseline_fss_values = self.fss_baseline[threshold][scale]

                # Calculate average FSS for this threshold and scale
                model_mean = np.nanmean(model_fss_values)
                baseline_mean = np.nanmean(baseline_fss_values)

                df_model.loc[threshold, f"scale_{scale}"] = model_mean
                df_baseline.loc[threshold, f"scale_{scale}"] = baseline_mean

                # Calculate improvement (can be negative if model is worse than baseline)
                improvement = model_mean - baseline_mean
                df_improvement.loc[threshold, f"scale_{scale}"] = improvement

        # Save DataFrames to CSV
        model_csv_file = os.path.join(fss_dir,
                                      f"dataset_{self.dataset_name}_ckpt_{checkpoint_name_cleaned}_fss_model.csv")
        baseline_csv_file = os.path.join(fss_dir,
                                         f"dataset_{self.dataset_name}_ckpt_{checkpoint_name_cleaned}_fss_baseline.csv")
        improvement_csv_file = os.path.join(fss_dir,
                                            f"dataset_{self.dataset_name}_ckpt_{checkpoint_name_cleaned}_fss_improvement.csv")

        df_model.to_csv(model_csv_file)
        df_baseline.to_csv(baseline_csv_file)
        df_improvement.to_csv(improvement_csv_file)

        print(f"Saved FSS evaluations to {fss_dir}")

        # Also save a combined DataFrame with all metrics
        df_all = pd.DataFrame()
        for threshold in self.thresholds:
            for scale in self.scales:
                key = f"threshold_{threshold}_scale_{scale}"
                df_all[f"{key}_model"] = [df_model.loc[threshold, f"scale_{scale}"]]
                df_all[f"{key}_baseline"] = [df_baseline.loc[threshold, f"scale_{scale}"]]
                df_all[f"{key}_improvement"] = [df_improvement.loc[threshold, f"scale_{scale}"]]

        combined_csv_file = os.path.join(fss_dir,
                                         f"dataset_{self.dataset_name}_ckpt_{checkpoint_name_cleaned}_fss_combined.csv")
        df_all.to_csv(combined_csv_file, index=False)


