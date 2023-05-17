import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from helper_functions import create_dilation_list
from modules_blocks import Network
import datetime
from load_data import img_one_hot, PrecipitationFilteredDataset, load_data_sequence_preliminary, normalize_data,\
    inverse_normalize_data, filtering_data_scraper, lognormalize_data, random_splitting_filtered_indecies
from torch.utils.data import Dataset, DataLoader

import numpy as np
from helper_functions import load_zipped_pickle, save_zipped_pickle, one_hot_to_mm, save_settings, save_whole_project
import os
from plotting.plot_img_histogram import plot_img_histogram
from plotting.plot_images import plot_target_vs_pred, plot_target_vs_pred_with_likelihood
from plotting.plot_quality_metrics import plot_mse_light, plot_mse_heavy, plot_losses, plot_average_preds, plot_pixelwise_preds
import warnings
from tests.test_basic_functions import test_all
from hurry.filesize import size
from tqdm import tqdm
import psutil

import lightning.pytorch as pl


class CNN(pl.LightningModule):
    def __init__(self, s_num_input_time_steps, s_upscale_c_to, s_num_bins_crossentropy, s_width_height, s_learning_rate,
                 device, s_width_height_target):
        self.model = Network(c_in=s_num_input_time_steps, s_upscale_c_to=s_upscale_c_to,
                s_num_bins_crossentropy=s_num_bins_crossentropy, s_width_height_in=s_width_height)
        self.model.to(device)

        self.s_learning_rate = s_learning_rate
        self.s_width_height_target = s_width_height_target

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.s_learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        input_sequence, target_one_hot, target = batch
        # Todo: get rid of float conversion? do this in filter already?
        input_sequence = input_sequence.float()
        target_one_hot = target_one_hot.float()
        # TODO targets already cropped??
        pred = self.model(input_sequence)
        loss = nn.CrossEntropyLoss(pred, target_one_hot)
        self.log('train_loss', loss)
        ### Additional quility metrics: ###
        # MSE

        # MSE persistence
        # persistence = input_sequence[:, -1, :, :]
        # persistence = T.CenterCrop(size=self.s_width_height_target)(persistence)

        return loss

    def validation_step(self, val_batch, batch_idx):
        input_sequence, target_one_hot, target = val_batch
        loss = nn.CrossEntropyLoss()



if __name__ == '__main__':
    #  Training data
    # num_training_samples = 20  # 1000  # Number of loaded pictures (first pics not used for training but only input)
    # num_validation_samples = 20  # 600

    # train_start_date_time = datetime.datetime(2020, 12, 1)
    # s_folder_path = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/weather_data/dwd_datensatz_bits/rv_recalc/RV_RECALC/hdf/'

    s_local_machine_mode = False

    s_sim_name_suffix = '_6_months_filter_all_below_0'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        import nvidia_smi
        nvidia_smi.nvmlInit()
    # device = 'cpu'

    if s_local_machine_mode:
        s_sim_name = 'Run_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        s_sim_name = 'Run_{}_ID_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), int(os.environ['SLURM_JOB_ID']))  # SLURM_ARRAY_TASK_ID

    s_dirs = {}
    s_dirs['save_dir'] = 'runs/{}{}'.format(s_sim_name, s_sim_name_suffix)
    s_dirs['plot_dir'] = '{}/plots'.format(s_dirs['save_dir'])
    s_dirs['plot_dir_images'] = '{}/images'.format(s_dirs['plot_dir'])
    s_dirs['model_dir'] = '{}/model'.format(s_dirs['save_dir'])
    s_dirs['code_dir'] = '{}/code'.format(s_dirs['save_dir'])

    for _, make_dir in s_dirs.items():
        if not os.path.exists(make_dir):
            os.makedirs(make_dir)

    settings =\
        {
            's_local_machine_mode': s_local_machine_mode,
            's_sim_name': s_sim_name,
            's_sim_same_suffix': s_sim_name_suffix,

            's_folder_path': '/mnt/qb/butz/bst981/weather_data/dwd_nc/rv_recalc_months/rv_recalc_months',
            's_data_file_names': ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(6)],  # ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(9)],# ['RV_recalc_data_2019-01.nc'], # ['RV_recalc_data_2019-01.nc', 'RV_recalc_data_2019-02.nc', 'RV_recalc_data_2019-03.nc'], #   # ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(9)],
            's_data_variable_name': 'RV_recalc',
            's_choose_time_span': False,
            's_time_span': (datetime.datetime(2020, 12, 1), datetime.datetime(2020, 12, 1)),
            's_ratio_training_data': 0.6,
            's_data_loader_chunk_size': 20,

            # Parameters that give the network architecture
            's_upscale_c_to': 32, #64, #128, # 512,
            's_num_bins_crossentropy': 64,

            # 'minutes_per_iteration': 5,
            's_width_height': 256,
            's_width_height_target': 32,
            's_learning_rate': 0.0001,  # Schedule this at some point??
            's_num_epochs': 1000,
            's_num_input_time_steps': 4,  # The number of subsequent time steps that are used for one predicition
            's_num_lead_time_steps': 1,  # 5, # The number of pictures that are skipped from last input time step to target, starts with 0
            's_optical_flow_input': False,  # Not yet working!
            's_batch_size': 55,  # batch size 22: Total: 32G, Free: 6G, Used:25G | Batch size 26: Total: 32G, Free: 1G, Used:30G --> vielfache von 8 am besten
            's_save_trained_model': True, # saves model every epoch
            's_load_model': False,
            's_load_model_name': 'Run_·20230220-191041',
            's_dirs': s_dirs,
            'device': device,

            # Log transform input/ validation data --> log binning --> log(x+1)
            's_log_transform': True,
            's_normalize': True,

            's_min_rain_ratio_target': 0.01, #Deactivated  # The minimal amount of rain required in the 32 x 32 target for target and its
            # prior input sequence to make it through the filter into the training data

            's_testing': True,

            # Plotting stuff
            's_no_plotting': False, # This sets all plotting boos below to False
            's_plot_average_preds_boo': True,
            's_plot_pixelwise_preds_boo': True,
            's_plot_target_vs_pred_boo': True,
            's_plot_mse_boo': True,
            's_plot_losses_boo': True,
            's_plot_img_histogram_boo': True,

        }

    if settings['s_no_plotting']:
        for en in ['s_plot_average_preds_boo', 's_plot_pixelwise_preds_boo', 's_plot_target_vs_pred_boo', 's_plot_mse_boo',
                   's_plot_losses_boo', 's_plot_img_histogram_boo']:
            settings[en] = False

    if settings['s_local_machine_mode']:
        # settings['s_data_variable_name'] = 'WN_forecast'
        settings['s_data_variable_name'] = 'RV_recalc'

        # settings['s_folder_path'] = 'dwd_nc/test_data'
        # settings['s_folder_path'] = '/mnt/common/Jan/Programming/weather_data/dwd_nc/rv_recalc_months'
        # settings['s_folder_path'] = '/mnt/common/Jan/Programming/first_CNN_on_Radolan/dwd_nc/own_test_data'
        # settings['s_folder_path'] = '/mnt/common/Jan/Programming/first_CNN_on_Radolan/dwd_nc/own_test_data'
        settings['s_folder_path'] = 'dwd_nc/own_test_data'

        # settings['s_data_file_names'] = ['DE1200_RV_Recalc_20190101.nc']
        # settings['s_data_file_names'] = ['RV_recalc_data_2019-01.nc']
        settings['s_data_file_names'] = ['RV_recalc_data_2019-01_subset.nc']
        # settings['s_data_file_names'] = ['RV_recalc_data_2019-01_subset_bigger.nc']

        # settings['s_choose_time_span'] = True
        settings['s_choose_time_span'] = False
        # settings['s_time_span'] = (datetime.datetime(2019, 1, 1, 0), datetime.datetime(2019, 1, 1, 5))
        settings['s_time_span'] = (67, 150)  # <-- now done according to index (isel instead of sel)
        settings['s_upscale_c_to'] = 32  # 8
        settings['s_batch_size'] = 2
        settings['s_data_loader_chunk_size'] = 2
        settings['s_testing'] = True  # Runs tests at the beginning
        settings['s_min_rain_ratio_target'] = 0  # Deactivated # No Filter
        # FILTER NOT WORKING YET, ALWAYS RETURNS TRUE FOR TEST PURPOSES!!

    main(settings=settings, **settings)