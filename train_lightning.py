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

import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import CSVLogger
from helper_functions import one_hot_to_mm
import mlflow


class Network_l(pl.LightningModule):
    def __init__(self, linspace_binning_params, device, s_num_input_time_steps, s_upscale_c_to, s_num_bins_crossentropy, s_width_height, s_learning_rate,
                 s_width_height_target, **__):
        super().__init__()
        self.model = Network(c_in=s_num_input_time_steps, s_upscale_c_to=s_upscale_c_to,
                s_num_bins_crossentropy=s_num_bins_crossentropy, s_width_height_in=s_width_height)
        self.model.to(device)

        self.s_learning_rate = s_learning_rate
        self.s_width_height_target = s_width_height_target

        self._linspace_binning_params = linspace_binning_params


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
        loss = nn.CrossEntropyLoss()(pred, target_one_hot)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        # MLFlow
        mlflow.log_metric('train_loss', loss.item(), step=batch_idx)
        ### Additional quality metrics: ###

        is_first_step = (batch_idx == 0)
        # if True:
        # linspace_binning_min, linspace_binning_max, linspace_binning = self._linspace_binning_params
        # pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1,
        #                         mean_bin_vals=True)
        # pred_mm = torch.tensor(pred_mm, device=self.device)
        #
        # # MSE
        # mse_pred_target = torch.nn.MSELoss()(pred_mm, target)
        # self.log('train_mse_pred_target', mse_pred_target.item())
        # # mlflow.log_metric('train_mse_pred_target', mse_pred_target.item(), step=batch_idx)
        #
        # # MSE zeros
        # mse_zeros_target= torch.nn.MSELoss()(torch.zeros(target.shape, device=self.device), target)
        # self.log('train_mse_zeros_target', mse_zeros_target)
        # # mlflow.log_metric('train_mse_zeros_target', mse_zeros_target.item(), step=batch_idx)
        #
        # persistence = input_sequence[:, -1, :, :]
        # persistence = T.CenterCrop(size=self.s_width_height_target)(persistence)
        # mse_persistence_target = torch.nn.MSELoss()(persistence, target)
        # self.log('train_mse_persistence_target', mse_persistence_target)
        # # mlflow.log_metric('train_mse_persistence_target', mse_persistence_target.item(), step=batch_idx)

        return loss

    def validation_step(self, val_batch, batch_idx):
        input_sequence, target_one_hot, target = val_batch

        input_sequence = input_sequence.float()
        target_one_hot = target_one_hot.float()

        pred = self.model(input_sequence)
        loss = nn.CrossEntropyLoss()(pred, target_one_hot)

        self.log('val_loss', loss, on_step=False, on_epoch=True)

        # pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)
        # pred_mm = torch.from_numpy(pred_mm).detach()

        # MSE
        # mse_pred_target = torch.nn.MSELoss()(pred, target)
        # self.log('val_mse_pred_target', mse_pred_target)

        # MSE zeros
        # mse_zeros_target= torch.nn.MSELoss()(torch.zeros(target.shape), target)
        # self.log('val_mse_zeros_target', mse_zeros_target)


class TrainingLogsCallback(pl.Callback):
    # Important info in:
    # /home/jan/.cache/JetBrains/PyCharm2022.3/remote_sources/1316491302/-638459852/pytorch_lightning/callbacks/callback.py
    # /home/jan/.cache/JetBrains/PyCharm2022.3/remote_sources/1316491302/-638459852/lightning_fabric/loggers/csv_logs.py
    # /home/jan/.cache/JetBrains/PyCharm2022.3/remote_sources/1316491302/-638459852/pytorch_lightning/trainer/trainer.py
    def __init__(self, train_logger):
        super().__init__()
        self.train_logger = train_logger

    def on_train_epoch_end(self, trainer, pl_module):
        all_logs = trainer.callback_metrics  # Alternatively: trainer.logged_metrics

        # trainer.callback_metrics= {}
        # There are both, trainer and validation metrics in callback_metrics (and logged_metrics as well )
        train_logs = {key: value for key, value in all_logs.items() if 'train_' in key}
        self.train_logger.log_metrics(train_logs) #, step=trainer.current_epoch)
        self.train_logger.save()


    def on_train_end(self, trainer, pl_module):
        # self.train_logger.finalize()
        self.train_logger.save()


class ValidationLogsCallback(pl.Callback):
    def __init__(self, val_logger):
        super().__init__()
        self.val_logger = val_logger

    def on_validation_epoch_end(self, trainer, pl_module):
        all_logs = trainer.callback_metrics
        # trainer.callback_metrics = {}
        val_logs = {key: value for key, value in all_logs.items() if 'val_' in key}
        self.val_logger.log_metrics(val_logs)
        self.val_logger.log_metrics(val_logs) #, step=trainer.current_epoch)
        self.val_logger.save()

    def on_validation_end(self, trainer, pl_module):
        # self.train_logger.finalize()
        self.val_logger.save()



def data_loading(transform_f, settings, s_ratio_training_data, s_num_input_time_steps, s_num_lead_time_steps, s_normalize,
                s_num_bins_crossentropy, s_data_loader_chunk_size, s_batch_size, s_num_workers_data_loader, **__):
    # relative index of last input picture (starting from first input picture as idx 1)
    last_input_rel_idx = s_num_input_time_steps
    #  relative index of target picture (starting from first input picture as idx 1)
    target_rel_idx = s_num_input_time_steps + 1 + s_num_lead_time_steps

    ###############
    # FILTER
    # Save all index chunks that passed filter in filtered_indecies together with normalization statistics and
    # linspace_binning

    filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized, linspace_binning_max_unnormalized =\
        filtering_data_scraper(transform_f=transform_f, last_input_rel_idx=last_input_rel_idx, target_rel_idx=target_rel_idx,
                               **settings)

    ###############
    # LINSPACE BINNING
    # Normalize linspace binning thresholds now that data is available
    linspace_binning_min = lognormalize_data(linspace_binning_min_unnormalized, mean_filtered_data, std_filtered_data,
                                             transform_f, s_normalize)
    # Subtract a small number to account for rounding errors made in the normalization process
    linspace_binning_max = lognormalize_data(linspace_binning_max_unnormalized, mean_filtered_data, std_filtered_data,
                                             transform_f, s_normalize)

    linspace_binning_min = linspace_binning_min - 0.1
    linspace_binning_max = linspace_binning_max + 0.1

    linspace_binning = np.linspace(linspace_binning_min, linspace_binning_max, num=s_num_bins_crossentropy,
                                   endpoint=False)  # num_indecies + 1 as the very last entry will never be used

    ###############

    # Defining and splitting into training and validation data set
    num_training_samples = int(len(filtered_indecies) * s_ratio_training_data)
    num_validation_samples = len(filtered_indecies) - num_training_samples

    filtered_indecies_training, filtered_indecies_validation = random_splitting_filtered_indecies(
        filtered_indecies, num_training_samples, num_validation_samples, s_data_loader_chunk_size)

    train_data_set = PrecipitationFilteredDataset(filtered_indecies_training, mean_filtered_data, std_filtered_data,
                                                  linspace_binning_min, linspace_binning_max, linspace_binning,
                                                  transform_f, **settings)

    validation_data_set = PrecipitationFilteredDataset(filtered_indecies_validation, mean_filtered_data,
                                                       std_filtered_data,
                                                       linspace_binning_min, linspace_binning_max, linspace_binning,
                                                       transform_f, **settings)

    train_data_loader = DataLoader(train_data_set, batch_size=s_batch_size, shuffle=True, drop_last=True,
                                   num_workers=s_num_workers_data_loader)
    del train_data_set

    validation_data_loader = DataLoader(validation_data_set, batch_size=s_batch_size, shuffle=False, drop_last=True,
                                        num_workers=s_num_workers_data_loader)
    del validation_data_set


    print('Size data set: {} \nof which training samples: {}  \nvalidation samples: {}'.format(len(filtered_indecies),
                                                                                                 num_training_samples,
                                                                                                 num_validation_samples))
    print('Num training batches: {} \nNum validation Batches: {} \nBatch size: {}'.format(len(train_data_loader),
                                                                                       len(validation_data_loader),
                                                                                       s_batch_size))
    linspace_binning_params = (linspace_binning_min, linspace_binning_max, linspace_binning)
    return train_data_loader, validation_data_loader, linspace_binning_params


def train_wrapper(settings, s_log_transform, s_dirs, s_model_every_n_epoch, s_profiling, s_max_epochs, **__):
    '''
    All the junk surrounding train goes in here
    '''

    save_settings(settings, s_dirs['save_dir'])
    save_whole_project(s_dirs['code_dir'])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=s_dirs['model_dir'],
                                                       filename='model_{epoch}_{val_loss:.2f}',
                                                       save_top_k=-1,
                                                       every_n_epochs=s_model_every_n_epoch)

    if s_profiling:
        profiler = PyTorchProfiler(dirpath=s_dirs['profile_dir'], export_to_chrome=True)
    else:
        profiler = None

    # save_top_k=-1, prevents callback from overwriting previous checkpoints

    if s_log_transform:
        transform_f = lambda x: np.log(x + 1)
    else:
        transform_f = lambda x: x

    train_data_loader, validation_data_loader, linspace_binning_params = \
        data_loading(transform_f, settings, **settings)

    train_logger = CSVLogger(s_dirs['logs'], name='train_log')
    val_logger = CSVLogger(s_dirs['logs'], name='val_log')
    logger = [train_logger, val_logger]

    callback_list = [checkpoint_callback,
                     TrainingLogsCallback(train_logger),
                     ValidationLogsCallback(val_logger)]

    train_l(train_data_loader, validation_data_loader, profiler, callback_list, s_max_epochs,
            linspace_binning_params, logger, settings)


def train_l(train_data_loader, validation_data_loader, profiler, callback_list, max_epochs, linspace_binning_params,
            logger, settings):
    '''
    Train loop, keep this clean!
    '''

    model_l = Network_l(linspace_binning_params, **settings)

    trainer = pl.Trainer(callbacks=callback_list, profiler=profiler, max_epochs=max_epochs, log_every_n_steps=1,
                         logger=False)
    # trainer.logger = logger
    trainer.fit(model_l, train_data_loader, validation_data_loader)


if __name__ == '__main__':

    #  Training data
    # num_training_samples = 20  # 1000  # Number of loaded pictures (first pics not used for training but only input)
    # num_validation_samples = 20  # 600

    # train_start_date_time = datetime.datetime(2020, 12, 1)
    # s_folder_path = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/weather_data/dwd_datensatz_bits/rv_recalc/RV_RECALC/hdf/'

    s_local_machine_mode = True

    s_sim_name_suffix = '_test_profiler'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        import nvidia_smi

        nvidia_smi.nvmlInit()
    # device = 'cpu'

    if s_local_machine_mode:
        s_sim_name = 'Run_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        s_sim_name = 'Run_{}_ID_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                           int(os.environ['SLURM_JOB_ID']))  # SLURM_ARRAY_TASK_ID

    s_dirs = {}
    s_dirs['save_dir'] = 'runs/{}{}'.format(s_sim_name, s_sim_name_suffix)
    s_dirs['plot_dir'] = '{}/plots'.format(s_dirs['save_dir'])
    s_dirs['plot_dir_images'] = '{}/images'.format(s_dirs['plot_dir'])
    s_dirs['model_dir'] = '{}/model'.format(s_dirs['save_dir'])
    s_dirs['code_dir'] = '{}/code'.format(s_dirs['save_dir'])
    s_dirs['profile_dir'] = '{}/profile'.format(s_dirs['save_dir'])
    s_dirs['logs'] = '{}/logs'.format(s_dirs['save_dir'])


    for _, make_dir in s_dirs.items():
        if not os.path.exists(make_dir):
            os.makedirs(make_dir)

    settings = \
        {
            's_local_machine_mode': s_local_machine_mode,
            's_sim_name': s_sim_name,
            's_sim_same_suffix': s_sim_name_suffix,

            's_max_epochs': None, # Max number of epochs, if None runs infenitely
            's_folder_path': '/mnt/qb/butz/bst981/weather_data/dwd_nc/rv_recalc_months/rv_recalc_months',
            's_data_file_names': ['RV_recalc_data_2019-0{}.nc'.format(i + 1) for i in range(6)],
            # ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(9)],# ['RV_recalc_data_2019-01.nc'], # ['RV_recalc_data_2019-01.nc', 'RV_recalc_data_2019-02.nc', 'RV_recalc_data_2019-03.nc'], #   # ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(9)],
            's_data_variable_name': 'RV_recalc',
            's_choose_time_span': False,
            's_time_span': (datetime.datetime(2020, 12, 1), datetime.datetime(2020, 12, 1)),
            's_ratio_training_data': 0.6,
            's_data_loader_chunk_size': 20, #  Chunk size, that consecutive data is chunked in when performing random splitting
            's_num_workers_data_loader': 4,

            # Parameters that give the network architecture
            's_upscale_c_to': 32,  # 64, #128, # 512,
            's_num_bins_crossentropy': 64,

            # 'minutes_per_iteration': 5,
            's_width_height': 256,
            's_width_height_target': 32,
            's_learning_rate': 0.0001,  # Schedule this at some point??
            's_num_epochs': 1000,
            's_num_input_time_steps': 4,  # The number of subsequent time steps that are used for one predicition
            's_num_lead_time_steps': 1,
            # 5, # The number of pictures that are skipped from last input time step to target, starts with 0
            's_optical_flow_input': False,  # Not yet working!
            's_batch_size': 55,
            # batch size 22: Total: 32G, Free: 6G, Used:25G | Batch size 26: Total: 32G, Free: 1G, Used:30G --> vielfache von 8 am besten
            's_save_trained_model': True,  # saves model every epoch
            's_load_model': False,
            's_load_model_name': 'Run_·20230220-191041',
            's_dirs': s_dirs,
            'device': device,

            # Log transform input/ validation data --> log binning --> log(x+1)
            's_log_transform': True,
            's_normalize': True,

            's_min_rain_ratio_target': 0.01,
            # Deactivated  # The minimal amount of rain required in the 32 x 32 target for target and its
            # prior input sequence to make it through the filter into the training data

            's_testing': True, # Runs tests before starting training
            's_profiling': False,  # Runs profiler
            # UNUSED!'s_calculate_quality_params': True, # Calculatiing quality params during training and validation

            # Plotting stuff
            's_no_plotting': False,  # This sets all plotting boos below to False
            's_plot_average_preds_boo': True,
            's_plot_pixelwise_preds_boo': True,
            's_plot_target_vs_pred_boo': True,
            's_plot_mse_boo': True,
            's_plot_losses_boo': True,
            's_plot_img_histogram_boo': True,

            # Logging Stuff
            's_model_every_n_epoch': 1, # Save model every nth epoch
            's_log_every_n_steps': 1, #Log argument that is passed to pl.Trainer. Trainer default is 50

        }

    if settings['s_no_plotting']:
        for en in ['s_plot_average_preds_boo', 's_plot_pixelwise_preds_boo', 's_plot_target_vs_pred_boo',
                   's_plot_mse_boo', 's_plot_losses_boo', 's_plot_img_histogram_boo']:
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
        settings['s_num_workers_data_loader'] = 0 # Debugging only works with zero workers
        settings['s_max_epochs'] = 2
        # FILTER NOT WORKING YET, ALWAYS RETURNS TRUE FOR TEST PURPOSES!!

    mlflow.create_experiment(settings['s_sim_name'])
    mlflow.pytorch.autolog()
    mlflow.log_models = False


    train_wrapper(settings, **settings)

