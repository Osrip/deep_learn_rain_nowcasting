
# create env conda create --name first_CNN_on_Radolan python=3.10

# Add default channels
#conda config --add channels conda-forge
#conda config --show channels

# install pytorch:
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install memory_profiler pytorch-lightning  mlflow ffmpeg matplotlib tqdm h5py xarray dask netCDF4 bottleneck -y

conda install einops pysteps -y
# conda install -c pytorch torchaudio
conda install -c bioconda hurry.filesize -y
conda install -c anaconda scipy -y

# I forgot thhe following in the previous installs!
# pip3 install nvidia-ml-py3


# Installing by hand did not work too well, however copying virtual env with this worked:
# conda env export > environment.yml
# conda env create -f environment.yml


#OLD
#mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge memory_profiler -c conda-forge pytorch-lightning -c bioconda hurry.filesize -c conda-forge mlflow -c conda-forge ffmpeg -c conda-forge pysteps
#mamba install matplotlib einops tqdm h5py xarray dask netCDF4 bottleneck
#pip3 install nvidia-ml-py3








# For Pycharm remote conda environment choose python3 location in remote conda env
# In Pycharm under Run/Debug configuration set Environment variables to LD_LIBRARY_PATH=/usr/local/cuda/lib64 (path to cuda) in order to enable cuda
# mamba install -c conda-forge einops

