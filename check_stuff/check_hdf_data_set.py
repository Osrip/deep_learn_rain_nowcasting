import h5py
import xarray as xr
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from helper_functions import bin_to_one_hot_index_linear
import datetime
from exceptions import CountException
from torch.utils.data import Dataset, DataLoader
import einops

folder_path = '/mnt/common/Jan/Programming/weather_data/dwd_nc/rv_recalc_months/RV_recalc_data_2019-01.nc'

data_dataset = xr.open_dataset(folder_path)

interrupt = False
i = 0

while not interrupt:
    i += 1
    data_set = data_dataset.isel(time=i)
    data = data_set['RV_recalc'].values
    data = data[0]
    data = np.array(T.CenterCrop(size=32)(torch.from_numpy(data)))
    if ((data != -1000000000.0) & (data != 0)).any():
        where = np.argwhere((data != -1000000000.0) & (data != 0))
        print(i)
        # print(where)
