import random
import time

import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import os
import xarray as xr  # Add this import for xarray
from pysteps import verification
from torch.utils.data import Subset, DataLoader

from helper.pre_process_target_input import one_hot_to_lognormed_mm, inverse_normalize_data
from helper.memory_logging import print_gpu_memory, print_ram_usage, format_duration
import torchvision.transforms as T

from load_data_xarray import FilteredDatasetXr


class FSSEvaluationCallback(pl.Callback):
    """
    Callback for calculating Fractions Skill Score (FSS) across different scales and thresholds.
    Preserves sample-level information instead of averaging across samples.
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
        super().__init__()
        self.scales = scales
        self.thresholds = thresholds
        self.linspace_binning_params = linspace_binning_params
        self.checkpoint_name = checkpoint_name
        self.dataset_name = dataset_name
        self.settings = settings

        # Get the FSS calculation method from pysteps
        self.fss_calc = verification.get_method("FSS")

        # Store all FSS values with sample ID
        self.fss_model_samples = []
        self.fss_baseline_samples = []
        self.sample_idx_counter = 0  # Track global sample index across batches

    def on_predict_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch,
            batch_idx: int,
            dataloader_idx: int = 0
    ):
        """Process batch results and calculate FSS for different scales and thresholds."""
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
        Calculate FSS for a batch across all scales and thresholds, preserving sample-level data.
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

            # Track global sample index
            global_sample_idx = self.sample_idx_counter
            self.sample_idx_counter += 1

            sample_model_data = {'sample_idx': global_sample_idx}
            sample_baseline_data = {'sample_idx': global_sample_idx}

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

                    # Add to sample data
                    key = f"threshold_{threshold}_scale_{scale}"
                    sample_model_data[key] = fss_model
                    sample_baseline_data[key] = fss_baseline

            # Store this sample's data
            self.fss_model_samples.append(sample_model_data)
            self.fss_baseline_samples.append(sample_baseline_data)

    def on_predict_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Save FSS evaluations at the end of the prediction epoch."""
        self.save_fss_evaluations()

    def save_fss_evaluations(self):
        """Save sample-level FSS evaluations to zarr format using xarray."""
        s_dirs = self.settings['s_dirs']
        log_dir = s_dirs['logs']
        evaluation_dir = os.path.join(log_dir, "evaluation")
        fss_dir = os.path.join(evaluation_dir, "fss")
        os.makedirs(fss_dir, exist_ok=True)

        checkpoint_name_cleaned = self.checkpoint_name.replace(".ckpt", "")

        # Prepare data
        n_samples = self.sample_idx_counter

        # Create 3D arrays [sample, threshold, scale]
        model_data = np.full((n_samples, len(self.thresholds), len(self.scales)), np.nan)
        baseline_data = np.full((n_samples, len(self.thresholds), len(self.scales)), np.nan)

        # Fill arrays with FSS values
        for sample_data in self.fss_model_samples:
            sample_idx = sample_data['sample_idx']
            for t_idx, threshold in enumerate(self.thresholds):
                for s_idx, scale in enumerate(self.scales):
                    key = f"threshold_{threshold}_scale_{scale}"
                    if key in sample_data:
                        model_data[sample_idx, t_idx, s_idx] = sample_data[key]

        for sample_data in self.fss_baseline_samples:
            sample_idx = sample_data['sample_idx']
            for t_idx, threshold in enumerate(self.thresholds):
                for s_idx, scale in enumerate(self.scales):
                    key = f"threshold_{threshold}_scale_{scale}"
                    if key in sample_data:
                        baseline_data[sample_idx, t_idx, s_idx] = sample_data[key]

        # Create xarray DataArrays
        model_da = xr.DataArray(
            data=model_data,
            dims=['sample', 'threshold', 'scale'],
            coords={
                'sample': np.arange(n_samples),
                'threshold': self.thresholds,
                'scale': self.scales
            },
            name='fss_model'
        )

        baseline_da = xr.DataArray(
            data=baseline_data,
            dims=['sample', 'threshold', 'scale'],
            coords={
                'sample': np.arange(n_samples),
                'threshold': self.thresholds,
                'scale': self.scales
            },
            name='fss_baseline'
        )

        # Combine into Dataset
        ds = xr.Dataset({
            'fss_model': model_da,
            'fss_baseline': baseline_da
        })

        # Add metadata
        ds.attrs['checkpoint_name'] = self.checkpoint_name
        ds.attrs['dataset_name'] = self.dataset_name
        ds.attrs['creation_time'] = str(pd.Timestamp.now())

        # Save to zarr format instead of NetCDF
        zarr_path = os.path.join(fss_dir,
                                 f"dataset_{self.dataset_name}_ckpt_{checkpoint_name_cleaned}_fss.zarr")
        ds.to_zarr(zarr_path, consolidated=True)  # consolidated=True improves access performance

        print(f"Saved FSS evaluations to {zarr_path}")