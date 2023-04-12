# pip3 install matplotlib torch einops tqdm h5py torchvision
# create env conda create --name first_CNN_on_Radolan python=3.10
mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
mamba install matplotlib einops tqdm h5py xarray dask netCDF4 bottleneck
mamba install -c bioconda hurry.filesize
pip3 install nvidia-ml-py3
# For Pycharm remote conda environment choose python3 location in remote conda env
# In Pycharm under Run/Debug configuration set Environment variables to LD_LIBRARY_PATH=/usr/local/cuda/lib64 (path to cuda) in order to enable cuda
# mamba install -c conda-forge einops