import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Full path to the zarr file
dem_path = '/home/jan/nowcasting_project/weather_data/static/dem_benchmark_dataset_1200_1100.zarr'

# Open the zarr file with xarray
try:
    ds = xr.open_zarr(dem_path)
    print(f"Successfully opened {dem_path} as xarray Dataset")
    print(f"Dataset dimensions: {ds.dims}")
    print(f"Available variables: {list(ds.data_vars)}")
except Exception as e:
    print(f"Error opening {dem_path}: {e}")
    exit(1)

# Access the dem variable
dem = ds.dem

# Get the dimensions
height, width = dem.shape
print(f"DEM shape: {height} x {width}")

# Plot a downsampled version of the full DEM to avoid memory issues
# Create a slice with stride to get a manageable resolution
downsample_factor = 100  # Use 1/100th of the points in each dimension
full_dem_downsampled = dem[::downsample_factor, ::downsample_factor].compute()

print(f"Downsampled DEM shape for full visualization: {full_dem_downsampled.shape}")

# Get min and max for colormap scaling (from downsampled data to be efficient)
vmin, vmax = np.nanmin(full_dem_downsampled), np.nanmax(full_dem_downsampled)
print(f"Elevation range (from downsampled data): {vmin} to {vmax}")

# Create a downsampled full resolution plot of the DEM
plt.figure(figsize=(10, 10))
plt.imshow(full_dem_downsampled, cmap='terrain', vmin=vmin, vmax=vmax)
plt.colorbar(label='Elevation (m)')
plt.title('Digital Elevation Model (Downsampled)')
plt.tight_layout()
plt.savefig('dem_full_resolution.png', dpi=300, bbox_inches='tight')
plt.close()

# Get random starting indices for the 32x32 patch
max_h = height - 32
max_w = width - 32

if max_h <= 0 or max_w <= 0:
    print("DEM is too small to extract a 32x32 patch")
    exit(1)

h_start = random.randint(0, max_h)
w_start = random.randint(0, max_w)

# Extract the 32x32 patch and compute it (load the data from disk/cache)
print(f"Extracting 32x32 patch from position ({h_start}, {w_start})...")
patch = dem[h_start:h_start+32, w_start:w_start+32].compute()

# Get actual min/max of the patch
patch_vmin, patch_vmax = np.nanmin(patch), np.nanmax(patch)
print(f"Patch elevation range: {patch_vmin} to {patch_vmax}")

# Create a plot of the patch with its own scaling for better detail
plt.figure(figsize=(6, 6))
plt.imshow(patch, cmap='terrain')
plt.colorbar(label='Elevation (m)')
plt.title(f'32x32 DEM Patch\nPosition: [{h_start}, {w_start}]')
plt.tight_layout()
plt.savefig('dem 32x32_example.png', dpi=300, bbox_inches='tight')
plt.close()

# Also create a version with global scaling for comparison (optional)
plt.figure(figsize=(6, 6))
plt.imshow(patch, cmap='terrain', vmin=vmin, vmax=vmax)
plt.colorbar(label='Elevation (m)')
plt.title(f'32x32 DEM Patch (Global Scale)\nPosition: [{h_start}, {w_start}]')
plt.tight_layout()
plt.savefig('dem 32x32_example_global_scale.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved images as:")
print("1. 'dem_full_resolution.png' (Downsampled full DEM)")
print("2. 'dem 32x32_example.png' (32x32 patch)")
print("3. 'dem 32x32_example_global_scale.png' (32x32 patch with global scale)")