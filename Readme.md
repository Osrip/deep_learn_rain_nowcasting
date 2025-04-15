# Environment Setup for Precipitation Nowcasting Project

This document provides instructions for setting up the conda/mamba environment for the precipitation nowcasting project.

## Environment Setup Instructions

These instructions create a Python 3.10 environment with all required dependencies, ensuring compatibility between conda-forge packages and pip-installed PyTorch.

```bash
# Create environment with Python 3.10
mamba create -n first_CNN_on_Radolan python=3.10 -y

# Activate the environment
mamba activate first_CNN_on_Radolan

# Configure to use conda-forge with strict priority
conda config --env --add channels conda-forge
# Note: If you see "Warning: 'conda-forge' already in 'channels' list, moving to the top"
# this is normal and indicates conda-forge is being correctly prioritized
conda config --env --set channel_priority strict

# Install dependencies from conda-forge
mamba install -y pytorch-lightning xarray zarr numpy matplotlib pandas scipy dask pyarrow psutil h5py pyyaml einops pysteps wandb hurry.filesize

# Install PyTorch and related packages via pip3 (Nightly Cuda 12.8)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Alternatively install torch with cuda 12.6
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Important Notes

- We use the conda-forge channel with strict priority to ensure consistency among packages
- PyTorch is installed via pip to get the latest CUDA 12.8 support
- All other packages are installed through mamba/conda to maintain environment stability

## Adding Additional Packages

If you need to install additional packages later, always use mamba to preserve compatibility:

```bash
mamba install -y package-name
```

This will respect the conda-forge channel priority you've established.

## Environment Activation

When returning to work on the project, activate the environment with:

```bash
mamba activate first_CNN_on_Radolan
```