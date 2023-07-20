import torch
import torch.nn as nn
import torchvision.transforms as T
# from modules_blocks import Network
from network_lightning import Network_l
import datetime
from load_data import PrecipitationFilteredDataset, filtering_data_scraper, lognormalize_data,\
    random_splitting_filtered_indecies, calc_class_frequencies, class_weights_per_sample
from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np
from helper.helper_functions import save_zipped_pickle, save_dict_pickle_csv,\
    save_tuple_pickle_csv, save_whole_project
import os

import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger
import mlflow
from plotting.plot_quality_metrics_from_log import plot_qualities_main
from plotting.plot_lr_scheduler import plot_lr_schedule
from calc_from_checkpoint import plot_images_outer
import copy
import warnings




class TrainingLogsCallback(pl.Callback):
    # Important info in:
    # pytorch_lightning/callbacks/callback.py
    # lightning_fabric/loggers/csv_logs.py
    # pytorch_lightning/trainer/trainer.py
    def __init__(self, train_logger):
        super().__init__()
        self.train_logger = train_logger

    def on_train_epoch_end(self, trainer, pl_module):
        # on_train_batch_end

        all_logs = trainer.callback_metrics  # Alternatively: trainer.logged_metrics

        # trainer.callback_metrics= {}
        # There are both, trainer and validation metrics in callback_metrics (and logged_metrics as well )
        train_logs = {key: value for key, value in all_logs.items() if 'train_' in key}
        self.train_logger.log_metrics(train_logs) # , epoch=trainer.current_epoch) #, step=trainer.current_epoch)
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
        self.val_logger.log_metrics(val_logs) # , epoch=trainer.current_epoch)
        # self.val_logger.log_metrics(val_logs) #, step=trainer.current_epoch)
        self.val_logger.save()

    def on_validation_end(self, trainer, pl_module):
        # self.val_logger.finalize()
        self.val_logger.save()


def data_loading(transform_f, settings, s_ratio_training_data, s_num_input_time_steps, s_num_lead_time_steps, s_normalize,
                s_num_bins_crossentropy, s_data_loader_chunk_size, s_batch_size, s_num_workers_data_loader, s_dirs, **__):
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

    filter_and_normalization_params = filtered_indecies, mean_filtered_data, std_filtered_data,\
        linspace_binning_min_unnormalized, linspace_binning_max_unnormalized

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

    class_weights_target, class_count_target, sample_num_target = calc_class_frequencies(filtered_indecies_training, linspace_binning,
                                                                                  mean_filtered_data, std_filtered_data,
                                                                                  transform_f, settings, normalize=True, **settings)

    target_mean_weights = class_weights_per_sample(filtered_indecies_training, class_weights_target, linspace_binning,
                                                   mean_filtered_data, std_filtered_data, transform_f, settings,
                                                   normalize=True)

    # TODO: RETURN filtered indecies instead of data set
    train_data_set = PrecipitationFilteredDataset(filtered_indecies_training, mean_filtered_data, std_filtered_data,
                                                  linspace_binning_min, linspace_binning_max, linspace_binning,
                                                  transform_f, **settings)

    validation_data_set = PrecipitationFilteredDataset(filtered_indecies_validation, mean_filtered_data,
                                                       std_filtered_data,
                                                       linspace_binning_min, linspace_binning_max, linspace_binning,
                                                       transform_f, **settings)

    training_steps_per_epoch = len(train_data_set)

    sampler = WeightedRandomSampler(weights=target_mean_weights, num_samples=training_steps_per_epoch, replacement=True)

    # Does this assume same order in weights as in data_set?? --> Seems so!
    # replacement=True allows for oversampling and in exchange not showing all samples each epoch
    # num_samples gives number of samples per epoch. Setting to len data_set forces sampler to not show all samples each epoch

    train_data_loader = DataLoader(train_data_set, sampler=sampler, batch_size=s_batch_size, drop_last=True,
                                   num_workers=s_num_workers_data_loader)


    validation_data_loader = DataLoader(validation_data_set, batch_size=s_batch_size, shuffle=False, drop_last=True,
                                        num_workers=s_num_workers_data_loader)


    print('Size data set: {} \nof which training samples: {}  \nvalidation samples: {}'.format(len(filtered_indecies),
                                                                                                 num_training_samples,
                                                                                                 num_validation_samples))
    print('Num training batches: {} \nNum validation Batches: {} \nBatch size: {}'.format(len(train_data_loader),
                                                                                       len(validation_data_loader),
                                                                                       s_batch_size))
    linspace_binning_params = (linspace_binning_min, linspace_binning_max, linspace_binning)
    # tODO: RETURN filtered indecies instead of data set
    return train_data_loader, validation_data_loader, filtered_indecies_training, filtered_indecies_validation,\
        linspace_binning_params, filter_and_normalization_params, training_steps_per_epoch
    # training_steps_per_epoch only needed for lr_schedule_plotting


def train_wrapper(settings, s_log_transform, s_dirs, s_model_every_n_epoch, s_profiling, s_max_epochs, s_num_gpus, s_sim_name, **__):
    '''
    All the junk surrounding train goes in here
    '''

    save_dict_pickle_csv(settings, s_dirs['data_dir'], 'settings')

    save_whole_project(s_dirs['code_dir'])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=s_dirs['model_dir'],
                                                       filename='model_{epoch:04d}_{val_loss:.2f}',
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

    train_data_loader, validation_data_loader, filtered_indecies_training, filtered_indecies_validation, linspace_binning_params, \
        filer_and_normalization_params, training_steps_per_epoch = data_loading(transform_f, settings, **settings)

    save_zipped_pickle('{}/filtered_indecies_training'.format(s_dirs['data_dir']), filtered_indecies_training)
    save_zipped_pickle('{}/filtered_indecies_validation'.format(s_dirs['data_dir']), filtered_indecies_validation)

    save_zipped_pickle('{}/filter_and_normalization_params'.format(s_dirs['data_dir']), filer_and_normalization_params)


    # Save linspace params
    save_tuple_pickle_csv(linspace_binning_params, s_dirs['data_dir'], 'linspace_binning_params')

    train_logger = CSVLogger(s_dirs['logs'], name='train_log')
    val_logger = CSVLogger(s_dirs['logs'], name='val_log')

    # Todo: implement lr schedule! Problem needs optimizer, that is in Network_l class but scheduler is needed for callback list that is needed for init of trainer
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR()

    logger = MLFlowLogger(experiment_name="Default", tracking_uri="file:./mlruns", run_name=s_sim_name, # tags={"mlflow.runName": settings['s_sim_name']},
                          log_model=False)


    callback_list = [checkpoint_callback,
                     TrainingLogsCallback(train_logger),
                     ValidationLogsCallback(val_logger)]

    model_l = train_l(train_data_loader, validation_data_loader, profiler, callback_list, logger, training_steps_per_epoch, s_max_epochs,
            linspace_binning_params, s_dirs['data_dir'], s_num_gpus, settings)

    # Network_l, training_steps_per_epoch is returned to be able to plot lr_scheduler
    return model_l, training_steps_per_epoch


def train_l(train_data_loader, validation_data_loader, profiler, callback_list, logger, training_steps_per_epoch, max_epochs, linspace_binning_params,
            data_dir, num_gpus, settings):
    '''
    Train loop, keep this clean!
    '''

    model_l = Network_l(linspace_binning_params, training_steps_per_epoch = training_steps_per_epoch, **settings)
    save_zipped_pickle('{}/Network_l_class'.format(data_dir), model_l)

    trainer = pl.Trainer(callbacks=callback_list, profiler=profiler, max_epochs=max_epochs, log_every_n_steps=1,
                         logger=logger, devices=num_gpus) #, precision='16-mixed'


    # trainer.logger = logger
    trainer.fit(model_l, train_data_loader, validation_data_loader)

    # Network_l is returned to be able to plot lr_scheduler
    return model_l


if __name__ == '__main__':

    #  Training data
    # num_training_samples = 20  # 1000  # Number of loaded pictures (first pics not used for training but only input)
    # num_validation_samples = 20  # 600

    # train_start_date_time = datetime.datetime(2020, 12, 1)
    # s_folder_path = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/weather_data/dwd_datensatz_bits/rv_recalc/RV_RECALC/hdf/'

    s_local_machine_mode = False

    s_sim_name_suffix = 'gaussian_blur_tried_fixing_device_stuff'

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

    s_sim_name = s_sim_name + s_sim_name_suffix

    s_dirs = {}
    s_dirs['save_dir'] = 'runs/{}'.format(s_sim_name)
    s_dirs['plot_dir'] = '{}/plots'.format(s_dirs['save_dir'])
    s_dirs['plot_dir_images'] = '{}/images'.format(s_dirs['plot_dir'])
    s_dirs['model_dir'] = '{}/model'.format(s_dirs['save_dir'])
    s_dirs['code_dir'] = '{}/code'.format(s_dirs['save_dir'])
    s_dirs['profile_dir'] = '{}/profile'.format(s_dirs['save_dir'])
    s_dirs['logs'] = '{}/logs'.format(s_dirs['save_dir'])
    s_dirs['data_dir'] = '{}/data'.format(s_dirs['save_dir'])


    for _, make_dir in s_dirs.items():
        if not os.path.exists(make_dir):
            os.makedirs(make_dir)

    settings = \
        {
            's_local_machine_mode': s_local_machine_mode,
            's_sim_name': s_sim_name,
            's_sim_same_suffix': s_sim_name_suffix,

            's_max_epochs': 50, # Max number of epochs, affects scheduler (if None runs infinitely, does not work with scheduler)
            's_folder_path': '/mnt/qb/butz/bst981/weather_data/dwd_nc/rv_recalc_months/rv_recalc_months',
            's_data_file_names': ['RV_recalc_data_2019-{:02d}.nc'.format(i + 1) for i in range(12)],
            # ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(9)],# ['RV_recalc_data_2019-01.nc'], # ['RV_recalc_data_2019-01.nc', 'RV_recalc_data_2019-02.nc', 'RV_recalc_data_2019-03.nc'], #   # ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(9)],
            's_data_variable_name': 'RV_recalc',
            's_choose_time_span': False,
            's_time_span': (datetime.datetime(2020, 12, 1), datetime.datetime(2020, 12, 1)),
            's_ratio_training_data': 0.6,
            's_data_loader_chunk_size': 20, #  Chunk size, that consecutive data is chunked in when performing random splitting
            's_num_workers_data_loader': 4,

            # Parameters related to lightning
            's_num_gpus': 4,

            # Parameters that give the network architecture
            's_upscale_c_to': 32,  # 64, #128, # 512,
            's_num_bins_crossentropy': 64,

            # 'minutes_per_iteration': 5,
            's_width_height': 256,
            's_width_height_target': 32,
            's_learning_rate': 0.0001,  # 0.0001 Schedule this at some point??
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
            's_gaussian_smoothing_target': True,

            # Log transform input/ validation data --> log binning --> log(x+1)
            's_log_transform': True,
            's_normalize': True,

            's_min_rain_ratio_target': 0.01,
            # Deactivated  # The minimal amount of rain required in the 32 x 32 target for target and its
            # prior input sequence to make it through the filter into the training data

            's_testing': True, # Runs tests before starting training
            's_profiling': False,  # Runs profiler
            's_calculate_quality_params': True, # Calculatiing quality params during training and validation

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
        settings['s_time_span'] = (67, 150)  # Only used when s_choose_time_span == True; now done according to index (isel instead of sel)
        settings['s_upscale_c_to'] = 32  # 8
        settings['s_batch_size'] = 2
        settings['s_data_loader_chunk_size'] = 2
        settings['s_testing'] = True  # Runs tests at the beginning
        settings['s_min_rain_ratio_target'] = 0  # Deactivated # No Filter
        settings['s_num_workers_data_loader'] = 0 # Debugging only works with zero workers
        settings['s_max_epochs'] = 3
        settings['s_num_gpus'] = 1
        # FILTER NOT WORKING YET, ALWAYS RETURNS TRUE FOR TEST PURPOSES!!

    # mlflow.create_experiment(settings['s_sim_name'])
    # mlflow.set_tag("mlflow.runName", settings['s_sim_name'])
    # mlflow.pytorch.autolog()
    # mlflow.log_models = False



    model_l, training_steps_per_epoch = train_wrapper(settings, **settings)

    plot_metrics_settings = {
        'ps_sim_name': settings['s_sim_name'], # TODO: Solve conflicting name convention
    }


    plot_images_settings ={
        'ps_runs_path': '{}/runs'.format(os.getcwd()),
        'ps_run_name': settings['s_sim_name'],
        'ps_device': settings['device'],
        'ps_checkpoint_name': None,  # If none take checkpoint of last epoch

    }

    plot_lr_schedule_settings = {
        'ps_sim_name': settings['s_sim_name'], # TODO: Solve conflicting name convention
    }

    plot_qualities_main(plot_metrics_settings, **plot_metrics_settings)

    # Deepcopy lr_scheduler to make sure steps in instance is not messed up
    # lr_scheduler = copy.deepcopy(model_l.lr_scheduler)
    plot_lr_schedule(model_l.lr_scheduler, training_steps_per_epoch, settings['s_max_epochs'],
                     **plot_lr_schedule_settings)

    # Some weird error occurs here ever since
    try:
        plot_images_outer(plot_images_settings, **plot_images_settings)
    except Exception:
        warnings.warn('Image plotting encountered error!')





