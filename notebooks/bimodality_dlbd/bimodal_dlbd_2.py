#!/usr/bin/env python3
"""
Bimodal DLBD Analysis Script

This script applies the Distance Learning By Distribution (DLBD) preprocessing technique
to precipitation radar data and analyzes the resulting distributions for bimodality,
keeping operations on GPU for maximum efficiency.
"""

# disable dynamo for debugging:
import helper.disable_dynamo_torch

import os
import torch
import numpy as np
import xarray as xr
import pandas as pd
from typing import Union, Tuple, Dict, List, Any
import einops
import gzip
import pickle
from datetime import datetime
import yaml
from torch.utils.data import Dataset, DataLoader
import time
import glob

# Import the DLBD function
from helper.dlbd import dlbd_target_pre_processing


def load_zipped_pickle(file):
    """Load compressed pickle file"""
    with gzip.GzipFile(file + '.pickle.pgz', 'rb') as f:
        data = pickle.load(f)
    return data


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def find_data_loader_vars_file(settings_dlbd):
    """Find the most recent data_loader_vars file that matches the pattern"""
    try:
        # Check if a specific file is provided
        if 'specific_data_loader_vars_file' in settings_dlbd and settings_dlbd['specific_data_loader_vars_file']:
            path = os.path.join(settings_dlbd['s_data_loader_vars_path'],
                                settings_dlbd['specific_data_loader_vars_file'])
            if os.path.exists(f'{path}.pickle.pgz'):
                return path

        # Try to find a file matching the pattern
        pattern = f"*_cluster_32_*_256_*_oversampling.pickle.pgz"
        files = glob.glob(os.path.join(settings_dlbd['s_data_loader_vars_path'], pattern))

        if not files:
            raise FileNotFoundError(f"No data loader vars files found matching pattern: {pattern}")

        # Sort by modification time (newest first)
        files.sort(key=os.path.getmtime, reverse=True)

        # Remove .pickle.pgz extension
        newest_file = files[0][:-11]
        print(f"Using the most recent data loader vars file: {os.path.basename(newest_file)}")
        return newest_file
    except Exception as e:
        print(f"Error finding data loader vars file: {e}")
        raise


def load_data_loader_vars(settings_dlbd):
    """Load the data loader variables containing linspace binning parameters"""
    try:
        path = find_data_loader_vars_file(settings_dlbd)
        print(f'Loading data loader vars from {path}')
        return load_zipped_pickle(path)
    except Exception as e:
        print(f"Error loading data loader vars: {e}")
        raise


class RadolanDataset(Dataset):
    """Dataset for loading Radolan precipitation data timestamps"""

    def __init__(self, dataset, variable_name):
        self.dataset = dataset
        self.variable_name = variable_name
        self.times = self.dataset.time.values

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        time = self.times[idx]
        # Select data for this timestamp
        data = self.dataset.sel(time=time)[self.variable_name].values
        # Convert to tensor
        data = torch.from_numpy(data).float()
        # Convert time to string to avoid issues with batching datetime64 objects
        time_str = str(time)
        return {'data': data, 'time': time_str, 'time_orig': time}


def custom_collate_fn(batch):
    """
    Custom collate function to handle numpy.datetime64 objects.

    Args:
        batch: List of samples from the dataset

    Returns:
        Properly collated batch
    """
    data_batch = torch.stack([item['data'] for item in batch])
    time_str_batch = [item['time'] for item in batch]
    time_orig_batch = [item['time_orig'] for item in batch]

    return {
        'data': data_batch,
        'time': time_str_batch,
        'time_orig': time_orig_batch
    }


def img_one_hot(data_arr: torch.Tensor, num_c: int, linspace_binning: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Convert data to one-hot encoded form based on binning.

    Args:
        data_arr: Input data tensor
        num_c: Number of bins
        linspace_binning: Bin edges

    Returns:
        One-hot encoded tensor
    """
    # Convert linspace_binning to tensor if it's a numpy array
    if isinstance(linspace_binning, np.ndarray):
        linspace_binning = torch.from_numpy(linspace_binning).to(data_arr.device)

    # Make data_arr contiguous before bucketize
    data_arr = data_arr.contiguous()

    # For some reason we need right = True here
    indices = torch.bucketize(data_arr, linspace_binning, right=True) - 1

    # Handle out-of-bounds indices
    indices = torch.clamp(indices, 0, num_c - 1)

    # Create one-hot encoding
    data_hot = torch.nn.functional.one_hot(indices.long(), num_c)

    # Handle NaNs --> set one-hot to zeros
    nan_mask = torch.isnan(data_arr)
    data_hot[nan_mask] = torch.zeros(num_c, dtype=torch.long).to(data_hot.device)

    return data_hot


def detect_bimodality(distribution: torch.Tensor,
                      threshold: float = 0.1,
                      min_peak_height: float = 0.05,
                      min_peak_prominence: float = 0.05) -> bool:
    """
    Enhanced algorithm to detect if a distribution is bimodal.

    Args:
        distribution: Tensor distribution [num_bins]
        threshold: Minimum normalized depth between peaks to consider bimodal
        min_peak_height: Minimum height of peaks relative to max value
        min_peak_prominence: Minimum prominence of peaks relative to max value

    Returns:
        bool: True if the distribution is bimodal
    """
    # Normalize the distribution
    if torch.max(distribution) > 0:
        normalized_dist = distribution / torch.max(distribution)
    else:
        return False  # No signal

    # Find peaks (local maxima) with a more robust approach
    # A point is a peak if strictly greater than both neighbors
    is_peak = torch.zeros_like(normalized_dist, dtype=torch.bool)

    # Handle interior points
    interior_peaks = (normalized_dist[1:-1] > normalized_dist[:-2]) & (normalized_dist[1:-1] > normalized_dist[2:])
    is_peak[1:-1] = interior_peaks

    # Handle endpoints
    # Left endpoint is peak if greater than its right neighbor
    if normalized_dist[0] > normalized_dist[1]:
        is_peak[0] = True
    # Right endpoint is peak if greater than its left neighbor
    if normalized_dist[-1] > normalized_dist[-2]:
        is_peak[-1] = True

    # Filter peaks by minimum height
    is_peak = is_peak & (normalized_dist > min_peak_height)

    # Count number of significant peaks
    peak_positions = torch.nonzero(is_peak).squeeze(-1)
    num_peaks = peak_positions.numel()

    if num_peaks < 2:
        return False

    # Calculate prominence of each peak
    # For each peak, find the lowest point (min value) to left and right
    # until encountering a higher peak or edge
    peak_prominences = []

    for i, peak_pos in enumerate(peak_positions):
        # Convert to int for slicing
        peak_pos = int(peak_pos.item())
        peak_val = normalized_dist[peak_pos]

        # Find lowest point to the left (until higher peak or edge)
        left_min = peak_val
        for j in range(peak_pos - 1, -1, -1):
            if normalized_dist[j] > peak_val:  # Higher peak, stop
                break
            left_min = min(left_min, normalized_dist[j])

        # Find lowest point to the right (until higher peak or edge)
        right_min = peak_val
        for j in range(peak_pos + 1, len(normalized_dist)):
            if normalized_dist[j] > peak_val:  # Higher peak, stop
                break
            right_min = min(right_min, normalized_dist[j])

        # Prominence is peak height minus highest minimum
        prominence = peak_val - max(left_min, right_min)
        peak_prominences.append((prominence, peak_val, peak_pos))

    # Sort peaks by prominence (descending)
    peak_prominences.sort(reverse=True)

    # We need at least 2 peaks with sufficient prominence
    if len(peak_prominences) < 2 or peak_prominences[1][0] < min_peak_prominence:
        return False

    # Get positions of the two most prominent peaks
    peak1_pos = peak_prominences[0][2]
    peak2_pos = peak_prominences[1][2]

    # Find minimum between peaks
    left_idx = min(peak1_pos, peak2_pos)
    right_idx = max(peak1_pos, peak2_pos)

    if left_idx == right_idx:  # Should never happen but check anyway
        return False

    between_peaks = normalized_dist[left_idx:right_idx + 1]
    min_val = torch.min(between_peaks)

    # Highest and second highest peak values
    peak1_val = normalized_dist[peak1_pos]
    peak2_val = normalized_dist[peak2_pos]

    # Calculate relative depths from each peak to the minimum
    relative_depth1 = (peak1_val - min_val) / peak1_val
    relative_depth2 = (peak2_val - min_val) / peak2_val

    return relative_depth1 > threshold and relative_depth2 > threshold


def analyze_batch_bimodality(blurred_one_hot: torch.Tensor,
                             nan_mask: torch.Tensor,
                             bimodality_params: Dict) -> Tuple[torch.Tensor, torch.Tensor, List]:
    """
    Analyze the bimodality of each pixel in a batch of distributions.

    Args:
        blurred_one_hot: Tensor [batch, num_bins, height, width] of DLBD-processed distributions
        nan_mask: Tensor [batch, height, width] of NaN locations in original data
        bimodality_params: Dictionary of parameters for bimodality detection

    Returns:
        Tuple of (bimodal_counts, non_bimodal_counts, sample_bimodal_dists)
    """
    batch_size, num_bins, height, width = blurred_one_hot.shape
    device = blurred_one_hot.device

    # Initialize count tensors on GPU
    bimodal_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
    non_bimodal_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Maximum number of distributions to sample per batch
    max_samples = bimodality_params.get('max_samples_per_batch', 50)

    # Lists to store sample distributions
    sample_bimodal_dists = []

    # Process each batch item
    for b in range(batch_size):
        # Create a valid pixels mask (not NaN)
        valid_mask = ~nan_mask[b]

        # Convert the valid mask to indices
        valid_coords = valid_mask.nonzero(as_tuple=True)
        n_valid = valid_coords[0].size(0)

        # If there are too many valid pixels, randomly sample a subset
        # to analyze more thoroughly for bimodality
        # TODO er sampled randomly 1000 Pixel !!
        sample_indices = None
        if n_valid > 1000:
            sample_size = min(1000, n_valid)
            perm = torch.randperm(n_valid, device=device)[:sample_size]
            sample_indices = (valid_coords[0][perm], valid_coords[1][perm])
        else:
            sample_indices = valid_coords

        # Analyze sampled pixels for bimodality

        for i in range(sample_indices[0].size(0)):
            h, w = sample_indices[0][i], sample_indices[1][i]

            # Get distribution for this pixel
            dist = blurred_one_hot[b, :, h, w]

            # Check if bimodal using improved detection
            is_bimodal = detect_bimodality(
                dist,
                threshold=bimodality_params['threshold'],
                min_peak_height=bimodality_params['min_peak_height'],
                min_peak_prominence=bimodality_params['min_peak_prominence']
            )

            if is_bimodal:
                bimodal_counts[b] += 1

                # Collect sample distributions (up to max_samples per batch)
                if len(sample_bimodal_dists) < max_samples:
                    sample_bimodal_dists.append({
                        'batch_idx': b,
                        'h': h.item(),
                        'w': w.item(),
                        'distribution': dist.detach().cpu().numpy()
                    })
            else:
                non_bimodal_counts[b] += 1

    return bimodal_counts, non_bimodal_counts, sample_bimodal_dists


def write_results_chunk(results, file_path, mode='a'):
    """Write a chunk of results to CSV, creating file with header if needed"""
    if not results:
        return

    file_exists = os.path.exists(file_path)
    df = pd.DataFrame(results)
    df.to_csv(file_path, mode=mode, header=not file_exists, index=False)


def process_data(settings_dlbd, linspace_binning_params):
    """Main processing function to analyze bimodality in DLBD-processed data"""
    # Extract binning parameters
    _, _, linspace_binning = linspace_binning_params
    num_bins = len(linspace_binning)

    print(f"Using {num_bins} bins for processing")
    print(f"Using sigma={settings_dlbd['s_sigma']} for DLBD")

    # Open dataset
    data_path = os.path.join(settings_dlbd['s_folder_path'], settings_dlbd['s_data_file_name'])
    print(f"Opening dataset from {data_path}")

    try:
        # Using chunks 'auto' for most efficient data loading
        dataset = xr.open_zarr(data_path, decode_timedelta=False, chunks='auto')
    except Exception as e:
        print(f"Error opening dataset: {e}")
        # Try without chunks specification
        dataset = xr.open_zarr(data_path, decode_timedelta=False)

    # Squeeze the step dimension
    dataset = dataset.squeeze()
    print(f"Dataset dimensions after squeeze: {list(dataset.dims.items())}")

    # Filter time if specified
    if settings_dlbd['s_crop_data_time_span'] is not None:
        start_time = np.datetime64(settings_dlbd['s_crop_data_time_span'][0])
        stop_time = np.datetime64(settings_dlbd['s_crop_data_time_span'][1])
        print(f"Filtering time range: {start_time} to {stop_time}")
        crop_slice = slice(start_time, stop_time)
        dataset = dataset.sel(time=crop_slice)
        print(f"Dataset contains {len(dataset.time)} time steps after filtering")

    # Create dataloader
    print("Creating dataloader...")
    radolan_dataset = RadolanDataset(dataset, settings_dlbd['s_data_variable_name'])
    dataloader = DataLoader(
        radolan_dataset,
        batch_size=settings_dlbd['s_batch_size'],
        shuffle=False,
        num_workers=settings_dlbd['s_num_workers'],
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=custom_collate_fn  # Use custom collate function
    )

    # Prepare output directory
    os.makedirs(settings_dlbd['s_output_dir'], exist_ok=True)

    # Prepare CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(settings_dlbd['s_output_dir'], f'bimodality_results_{timestamp}.csv')
    bimodal_distributions_file = os.path.join(settings_dlbd['s_output_dir'], f'bimodal_distributions_{timestamp}.csv')

    # Save settings for reference
    with open(os.path.join(settings_dlbd['s_output_dir'], f'settings_{timestamp}.yaml'), 'w') as f:
        yaml.dump(settings_dlbd, f, default_flow_style=False)

    # Initialize counters
    total_bimodal_pixels = 0
    total_nonbimodal_pixels = 0
    total_processed_batches = 0
    total_processed_samples = 0

    # Initialize counters for threshold values
    total_pixels_exceeding_0_2mm = 0
    total_pixels_exceeding_1mm = 0
    total_pixels_exceeding_5mm = 0

    # Bimodality detection parameters
    bimodality_params = {
        'threshold': settings_dlbd['s_bimodality_threshold'],
        'min_peak_height': settings_dlbd['s_min_peak_height'],
        'min_peak_prominence': settings_dlbd['s_min_peak_prominence'],
        'max_samples_per_batch': settings_dlbd['s_max_samples_per_batch']
    }

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() and settings_dlbd['s_use_gpu'] else 'cpu')
    print(f"Using device: {device}")

    # Convert linspace_binning to tensor and move to device
    linspace_binning_tensor = torch.tensor(linspace_binning, device=device)

    # Timer for performance monitoring
    start_time = time.time()
    batch_start_time = start_time

    # Define precipitation thresholds in mm/h
    precip_thresholds = [0.2, 1.0, 5.0]

    print(f"Starting processing with batch size {settings_dlbd['s_batch_size']}...")
    try:
        # Process batches
        for batch_idx, batch in enumerate(dataloader):
            current_batch_size = batch['data'].shape[0]
            data = batch['data'].to(device)  # Shape: [batch_size, height, width]
            times = batch['time']  # Shape: [batch_size]
            times_orig = batch['time_orig']  # Original datetime64 objects for later use

            # Create nan mask
            nan_mask = torch.isnan(data)

            # Count pixels exceeding thresholds (excluding NaNs)
            pixels_exceeding_thresholds = []
            for threshold in precip_thresholds:
                # Calculate valid pixels that exceed the threshold (not NaN AND value > threshold)
                valid_exceeding = torch.logical_and(~nan_mask, data > threshold)
                # Sum over height and width dimensions for each batch item
                counts = valid_exceeding.sum(dim=(1, 2)).cpu().numpy()
                pixels_exceeding_thresholds.append(counts)

            # Handle NaNs for one-hot conversion
            data_for_onehot = torch.nan_to_num(data, nan=0.0)

            # Convert to one-hot
            data_one_hot = img_one_hot(data_for_onehot, num_bins, linspace_binning_tensor)
            # Rearrange dimensions to [batch, bins, height, width]
            data_one_hot = einops.rearrange(data_one_hot, 'b h w c -> b c h w')

            # Apply DLBD preprocessing
            sigma = settings_dlbd['s_sigma']
            kernel_size = settings_dlbd['s_kernel_size']

            # The negative padding determines by how much the dlbd processed output size is smaller than the input
            # this is determined by the kernel size, as the kernel reduces the output size

            # Kernel size has to be odd
            if kernel_size % 2 == 1:
                negative_padding = (kernel_size - 1) // 2
            else:
                raise ValueError('The kernel size has to be odd')

            # Get output dimensions (may not be square)
            output_h, output_w = data.shape[1] - negative_padding, data.shape[2] - negative_padding

            # Apply DLBD with same output size as input
            blurred_one_hot = dlbd_target_pre_processing(
                input_tensor=data_one_hot,
                output_size=(output_h, output_w),
                sigma=sigma,
                kernel_size=None
            )

            # Free memory
            del data_one_hot
            torch.cuda.empty_cache()

            # Find bimodal distributions using improved detection
            bimodal_counts, non_bimodal_counts, batch_bimodal_dists = analyze_batch_bimodality(
                blurred_one_hot,
                nan_mask,
                bimodality_params
            )

            # Free memory
            del blurred_one_hot
            torch.cuda.empty_cache()

            # Move counts to CPU for processing
            bimodal_counts = bimodal_counts.cpu().numpy()
            non_bimodal_counts = non_bimodal_counts.cpu().numpy()

            # Create batch results
            batch_results = []
            for b in range(current_batch_size):
                # Use time string stored in batch
                time_str = times[b]
                batch_results.append({
                    'time': time_str,
                    'num_pixels_bimodal': int(bimodal_counts[b]),
                    'num_pixels_not_bimodal': int(non_bimodal_counts[b]),
                    'num_pixels_exceeding_0_2mm': int(pixels_exceeding_thresholds[0][b]),
                    'num_pixels_exceeding_1mm': int(pixels_exceeding_thresholds[1][b]),
                    'num_pixels_exceeding_5mm': int(pixels_exceeding_thresholds[2][b])
                })

            # Stream results to CSV instead of keeping in memory
            write_results_chunk(batch_results, results_file)

            # Create bimodal distribution samples
            batch_bimodal_dist_records = []
            for i, dist_data in enumerate(batch_bimodal_dists):
                b = dist_data['batch_idx']
                time_str = times[b]
                batch_bimodal_dist_records.append({
                    'time': time_str,
                    'h': dist_data['h'],
                    'w': dist_data['w'],
                    **{f'bin_{j}': float(val) for j, val in enumerate(dist_data['distribution'])}
                })

            # Stream bimodal distributions to CSV
            write_results_chunk(batch_bimodal_dist_records, bimodal_distributions_file)

            # Update totals
            batch_bimodal_sum = bimodal_counts.sum()
            batch_nonbimodal_sum = non_bimodal_counts.sum()
            total_bimodal_pixels += batch_bimodal_sum
            total_nonbimodal_pixels += batch_nonbimodal_sum

            # Update threshold pixel counts
            total_pixels_exceeding_0_2mm += pixels_exceeding_thresholds[0].sum()
            total_pixels_exceeding_1mm += pixels_exceeding_thresholds[1].sum()
            total_pixels_exceeding_5mm += pixels_exceeding_thresholds[2].sum()

            total_processed_batches += 1
            total_processed_samples += current_batch_size

            # Report progress every n batches
            if total_processed_batches % settings_dlbd['s_report_every_n_batches'] == 0:
                # Calculate timing
                current_time = time.time()
                elapsed_batch_time = current_time - batch_start_time
                samples_per_second = (settings_dlbd['s_report_every_n_batches'] * settings_dlbd[
                    's_batch_size']) / elapsed_batch_time

                print(f"Processed {total_processed_samples} samples ({total_processed_batches} batches)")
                print(
                    f"Last {settings_dlbd['s_report_every_n_batches']} batches took {elapsed_batch_time:.2f}s ({samples_per_second:.2f} samples/s)")
                print(f"Total bimodal pixels: {total_bimodal_pixels}")
                print(f"Total non-bimodal pixels: {total_nonbimodal_pixels}")
                print(f"Total pixels exceeding 0.2mm/h: {total_pixels_exceeding_0_2mm}")
                print(f"Total pixels exceeding 1mm/h: {total_pixels_exceeding_1mm}")
                print(f"Total pixels exceeding 5mm/h: {total_pixels_exceeding_5mm}")

                # If any bimodal pixels found, report percentage
                if total_bimodal_pixels + total_nonbimodal_pixels > 0:
                    bimodal_percentage = 100 * total_bimodal_pixels / (total_bimodal_pixels + total_nonbimodal_pixels)
                    print(f"Bimodal percentage: {bimodal_percentage:.2f}%")

                # Reset batch timer
                batch_start_time = current_time

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Calculate total processing time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print("Processing complete or interrupted.")
        print(f"Total processing time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print(f"Total bimodal pixels: {total_bimodal_pixels}")
        print(f"Total non-bimodal pixels: {total_nonbimodal_pixels}")
        print(f"Total pixels exceeding 0.2mm/h: {total_pixels_exceeding_0_2mm}")
        print(f"Total pixels exceeding 1mm/h: {total_pixels_exceeding_1mm}")
        print(f"Total pixels exceeding 5mm/h: {total_pixels_exceeding_5mm}")

        # If any bimodal pixels found, report percentage
        if total_bimodal_pixels + total_nonbimodal_pixels > 0:
            bimodal_percentage = 100 * total_bimodal_pixels / (total_bimodal_pixels + total_nonbimodal_pixels)
            print(f"Bimodal percentage: {bimodal_percentage:.2f}%")

        print(f"Results saved to {settings_dlbd['s_output_dir']}")

def main():
    # Default config file path
    config_path = "config_dlbd.yml"

    # Check if alternative config file specified
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Creating default config file...")

        # Create default config
        default_config = {
            # Paths and data settings
            's_folder_path': "/home/butz/bst981/nowcasting_project/weather_data/radolan",
            's_data_file_name': "RV_recalc_rechunked.zarr",
            's_data_variable_name': "RV_recalc",
            's_crop_data_time_span': ["2019-01-01T00:00", "2019-02-01T00:00"],  # Smaller default window
            's_data_loader_vars_path': "/home/butz/bst981/nowcasting_project/output/data_loader_vars",
            'specific_data_loader_vars_file': None,  # Set to filename if you want a specific file

            # Processing settings
            's_batch_size': 4,
            's_kernel_size': 33,  # Should be odd
            's_sigma': 1.0,
            's_bimodality_threshold': 0.1,  # Sensitivity for bimodal detection
            's_min_peak_height': 0.05,  # Minimum normalized peak height
            's_min_peak_prominence': 0.05,  # Minimum normalized peak prominence
            's_max_samples_per_batch': 50,  # Max bimodal distributions to sample per batch
            's_report_every_n_batches': 10,
            's_output_dir': "./dlbd_results",
            's_num_workers': 4,  # DataLoader workers
            's_use_gpu': True
        }

        # Save default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        print(f"Default config written to {config_path}")
        print("Please review and edit this file, then run the script again.")
        return 0

    # Load config
    try:
        settings_dlbd = load_config(config_path)

        # Print settings
        print("Using the following settings:")
        for key, value in settings_dlbd.items():
            print(f"  {key}: {value}")

        # Validate critical settings
        if not os.path.exists(settings_dlbd['s_folder_path']):
            raise ValueError(f"Folder path does not exist: {settings_dlbd['s_folder_path']}")

        # Create output directory
        os.makedirs(settings_dlbd['s_output_dir'], exist_ok=True)

        # Load data_loader_vars
        data_loader_vars = load_data_loader_vars(settings_dlbd)

        # The linspace_binning_params is normally the 10th item in the data_loader_vars tuple
        try:
            linspace_binning_params = data_loader_vars[8]
        except (IndexError, TypeError):
            # Try to see if data_loader_vars itself is the linspace_binning_params
            if isinstance(data_loader_vars, tuple) and len(data_loader_vars) == 3:
                linspace_binning_params = data_loader_vars
                print("Using data_loader_vars directly as linspace_binning_params")
            else:
                raise ValueError("Could not extract linspace_binning_params from data_loader_vars")

        # Process the data
        process_data(settings_dlbd, linspace_binning_params)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()