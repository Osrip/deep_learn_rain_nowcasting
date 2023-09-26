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
from logger import ValidationLogsCallback, TrainingLogsCallback, BaselineTrainingLogsCallback, BaselineValidationLogsCallback,\
    create_loggers
from baselines import LKBaseline
import mlflow
from plotting.plot_quality_metrics_from_log import plot_qualities_main, plot_precipitation_diff
from plotting.plot_lr_scheduler import plot_lr_schedule, plot_sigma_schedule
from helper.sigma_scheduler_helper import create_scheduler_mapping
from helper.helper_functions import no_special_characters
from calc_from_checkpoint import plot_images_outer
import copy
import warnings


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
                                   num_workers=s_num_workers_data_loader, pin_memory=True)


    validation_data_loader = DataLoader(validation_data_set, batch_size=s_batch_size, shuffle=False, drop_last=True,
                                        num_workers=s_num_workers_data_loader, pin_memory=True)


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

def calc_baselines(data_loader_list, logs_callback_list, logger_list, logging_type_list, settings, **__):
    '''
    Goes into train_wrapper
    data_loader_list, logs_callback_list, logger_list, logging_type_list have to be in according order
    logging_type depending on data loader either: 'train' or 'val'
    '''


    for data_loader, logs_callback, logger, logging_type in \
            zip(data_loader_list, logs_callback_list, logger_list, logging_type_list):
        # Create callback list in the form of [BaselineTrainingLogsCallback(base_train_logger)]
        callback_list_base = [logs_callback(logger)]


        lk_baseline = LKBaseline(logging_type, **settings)

        trainer = pl.Trainer(callbacks=callback_list_base, max_epochs=1, log_every_n_steps=1, check_val_every_n_epoch=1)
        trainer.validate(lk_baseline, data_loader)


def train_wrapper(settings, s_log_transform, s_dirs, s_model_every_n_epoch, s_profiling, s_max_epochs, s_num_gpus,
                  s_sim_name, s_gaussian_smoothing_target, s_sigma_target_smoothing, s_schedule_sigma_smoothing,
                  s_check_val_every_n_epoch, s_calc_baseline, **__):
    '''
    All the junk surrounding train goes in here
    '''
    train_logger, val_logger, base_train_logger, base_val_logger = create_loggers(**settings)

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




    # eNABLE MLFLOW LOGGING HERE!
    # logger = MLFlowLogger(experiment_name="Default", tracking_uri="file:./mlruns", run_name=s_sim_name, # tags={"mlflow.runName": settings['s_sim_name']},
    #                       log_model=False)
    logger = None


    callback_list = [checkpoint_callback,
                     TrainingLogsCallback(train_logger),
                     ValidationLogsCallback(val_logger)]

    if s_gaussian_smoothing_target and s_schedule_sigma_smoothing:

        sigma_schedule_mapping, sigma_scheduler = create_scheduler_mapping(training_steps_per_epoch, s_max_epochs, s_sigma_target_smoothing)
    else:
        sigma_schedule_mapping, sigma_scheduler = (None, None)

    model_l = train_l(train_data_loader, validation_data_loader, profiler, callback_list, logger, training_steps_per_epoch, s_max_epochs,
            linspace_binning_params, s_dirs['data_dir'], s_num_gpus, sigma_schedule_mapping, s_check_val_every_n_epoch,
                      settings)

    if s_calc_baseline:
        calc_baselines(data_loader_list=[train_data_loader, validation_data_loader],
                       logs_callback_list=[BaselineTrainingLogsCallback, BaselineValidationLogsCallback],
                       logger_list=[base_train_logger, base_val_logger],
                       logging_type_list=['train', 'val'],
                       settings=settings)

    # Network_l, training_steps_per_epoch is returned to be able to plot lr_scheduler
    return model_l, training_steps_per_epoch, sigma_schedule_mapping


def train_l(train_data_loader, validation_data_loader, profiler, callback_list, logger, training_steps_per_epoch,
            max_epochs, linspace_binning_params, data_dir, num_gpus, sigma_schedule_mapping, check_val_every_n_epoch, settings):
    '''
    Train loop, keep this clean!
    '''

    model_l = Network_l(linspace_binning_params, sigma_schedule_mapping, training_steps_per_epoch = training_steps_per_epoch, **settings)
    save_zipped_pickle('{}/Network_l_class'.format(data_dir), model_l)

    trainer = pl.Trainer(callbacks=callback_list, profiler=profiler, max_epochs=max_epochs, log_every_n_steps=1,
                         logger=logger, devices=num_gpus, strategy="ddp", check_val_every_n_epoch=check_val_every_n_epoch) # precision='16-mixed'
    # Speed up advice: https://pytorch-lightning.readthedocs.io/en/1.8.6/guides/speed.html


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

    s_local_machine_mode = True

    s_sim_name_suffix = 'TEST_NO_gaussian_WITH_exp_lr_schedule' #'exp_sigma_schedule_no_lr_schedule' # 'No_Gaussian_blurring_with_lr_schedule_64_bins' #'sigma_init_5_exp_sigma_schedule_WITH_lr_schedule_xentropy_loss_20_min_lead_time'#'scheduled_sigma_exp_init_50_no_lr_schedule_100G_mem' #'sigma_50_no_sigma_schedule_no_lr_schedule' #'scheduled_sigma_exp_init_50_no_lr_schedule_100G_mem'# 'sigma_50_no_sigma_schedule_lr_init_0_001' # 'scheduled_sigma_exp_init_50_lr_init_0_001' #'no_gaussian_smoothing_lr_init_0_001' #'' #'scheduled_sigma_exp_init_50_lr_init_0_001' #'no_gaussian_smoothing_lr_init_0_001' #'scheduled_sigma_cos_init_20_to_0_1_lr_init_0_001' #'smoothing_constant_sigma_1_and_lr_schedule' #'scheduled_sigma_cos_init_20_to_0_1_lr_init_0_001'

    # Getting rid of all special characters except underscores
    s_sim_name_suffix = no_special_characters(s_sim_name_suffix)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if device.type == 'cuda':
    #     import nvidia_smi
    #
    #     nvidia_smi.nvmlInit()


    if s_local_machine_mode:
        s_sim_name = 'Run_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        s_sim_name = 'Run_{}_ID_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                           int(os.environ['SLURM_JOB_ID']))  # SLURM_ARRAY_TASK_ID

    s_sim_name = s_sim_name + s_sim_name_suffix

    s_dirs = {}
    if s_local_machine_mode:
        s_dirs['save_dir'] = 'runs/{}'.format(s_sim_name)
    else:
        s_dirs['save_dir'] = '/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs/{}'.format(s_sim_name)

    # s_dirs['save_dir'] = 'runs/{}'.format(s_sim_name)
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

            's_max_epochs': 50, # Max number of epochs, affects scheduler (if None: runs infinitely, does not work with scheduler)
            's_folder_path': '/mnt/qb/butz/bst981/weather_data/dwd_nc/rv_recalc_months/rv_recalc_months',
            's_data_file_names': ['RV_recalc_data_2019-{:02d}.nc'.format(i + 1) for i in range(12)],
            # ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(9)],# ['RV_recalc_data_2019-01.nc'], # ['RV_recalc_data_2019-01.nc', 'RV_recalc_data_2019-02.nc', 'RV_recalc_data_2019-03.nc'], #   # ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(9)],
            's_data_variable_name': 'RV_recalc',
            's_choose_time_span': False,
            's_time_span': (datetime.datetime(2020, 12, 1), datetime.datetime(2020, 12, 1)),
            's_ratio_training_data': 0.6,
            's_data_loader_chunk_size': 20, #  Chunk size, that consecutive data is chunked in when performing random splitting
            's_num_workers_data_loader': 8, # Should correspond to number of cpus, also increases cpu ram
            's_check_val_every_n_epoch': 1, # Calculate validation every nth epoch for speed up, NOT SURE WHETHER PLOTTING CAN DEAL WITH THIS BEING LARGER THAN 1 !!

            # Parameters related to lightning
            's_num_gpus': 4,

            # Parameters that give the network architecture
            's_upscale_c_to': 32,  # 64, #128, # 512,
            's_num_bins_crossentropy': 64, #256,

            # 'minutes_per_iteration': 5,
            's_width_height': 256,
            's_width_height_target': 32,
            's_num_epochs': 1000,
            's_num_input_time_steps': 4,  # The number of subsequent time steps that are used for one predicition
            's_num_lead_time_steps': 3, # 0 --> 0 min prediction (target == last input) ; 1 --> 5 min predicition, 10 --> 15min etc
            # This is substracted by 2: settings['s_num_lead_time_steps'] = 's_num_lead_time_steps' -2 for following reasons:
            # 5, # The number of pictures that are skipped from last input time step to target, starts with -1
            # (starts counting at filtered_data_loader_indecies_dict['last_idx_input_sequence'], where last index is excess
            # for arange ((np.arange(1:5) = [1,2,3,4])
            's_optical_flow_input': False,  # Not yet working!
            's_batch_size': 45, # 55, downgraded to 45 after memory issue on v100 with smoothing stuff
            # batch size 22: Total: 32G, Free: 6G, Used:25G | Batch size 26: Total: 32G, Free: 1G, Used:30G --> vielfache von 8 am besten
            's_save_trained_model': True,  # saves model every epoch
            's_load_model': False,
            's_load_model_name': 'Run_·20230220-191041',
            's_dirs': s_dirs,
            'device': device,
            's_learning_rate': 0.001,  # 0.0001
            's_lr_schedule': True,  # enables lr scheduler, takes s_learning_rate as initial rate

            # Gaussian smoothing
            's_gaussian_smoothing_target': False,
            's_sigma_target_smoothing': 5,  # In case of scheduling this is the initial sigma
            's_schedule_sigma_smoothing': False,
            's_gaussian_smoothing_multiple_sigmas': False, # ignores s_gaussian_smoothing_target, s_sigma_target_smoothing and s_schedule_sigma_smoothing
            's_multiple_sigmas': [2, 4, 8, 12], # List of sigmas in case s_gaussian_smoothing_multiple_sigmas == True; to create loss mean is taken of all losses that each single sigma would reate

            # Logging
            's_calc_baseline': True, # Baselines are calculated and plotted
            's_log_precipitation_difference': True,
            's_calculate_quality_params': True, # Calculatiing quality params during training and validation
            's_calculate_fss': True, # Calculating fractions skill score during training and validation
            's_fss_scales': [2, 4, 8, 12], #[2, 16, 32], # Scales for which fss is calculated as a list
            's_fss_threshold': 1, # Threshold in mm/h for which fss is calculated

            # Log transform input/ validation data --> log binning --> log(x+1)
            's_log_transform': True,
            's_normalize': True,

            's_min_rain_ratio_target': 0.01,
            # Deactivated  # The minimal amount of rain required in the 32 x 32 target for target and its
            # prior input sequence to make it through the filter into the training data

            's_testing': True, # Runs tests before starting training
            's_profiling': False,  # Runs profiler

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
        # settings['s_folder_path'] = '/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/dwd_nc/own_test_data' # activate this for cluster

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
        settings['s_max_epochs'] = 1 # 3
        settings['s_num_gpus'] = 1
        # FILTER NOT WORKING YET, ALWAYS RETURNS TRUE FOR TEST PURPOSES!!

    settings['s_num_lead_time_steps'] = settings['s_num_lead_time_steps'] - 2

    # mlflow.create_experiment(settings['s_sim_name'])
    # mlflow.set_tag("mlflow.runName", settings['s_sim_name'])
    # mlflow.pytorch.autolog()
    # mlflow.log_models = False

    model_l, training_steps_per_epoch, sigma_schedule_mapping = train_wrapper(settings, **settings)

    plot_metrics_settings = {
        'ps_sim_name': s_dirs['save_dir'] # settings['s_sim_name']
    }


    plot_qualities_main(plot_metrics_settings, **plot_metrics_settings, **settings)

    if settings['s_log_precipitation_difference']:
        plot_precipitation_diff(plot_metrics_settings, **plot_metrics_settings, **settings)

    # Deepcopy lr_scheduler to make sure steps in instance is not messed up
    # lr_scheduler = copy.deepcopy(model_l.lr_scheduler)

    plot_lr_schedule_settings = {
        'ps_sim_name': s_dirs['save_dir'] # settings['s_sim_name'], # TODO: Solve conflicting name convention
    }


    if settings['s_lr_schedule']:

        plot_lr_schedule(model_l.lr_scheduler, training_steps_per_epoch, settings['s_max_epochs'],
                         save_name='lr_scheduler', y_label='Learning Rate', title='LR scheduler',
                         ylog=True, **plot_lr_schedule_settings)

    if settings['s_schedule_sigma_smoothing']:
        plot_sigma_schedule(sigma_schedule_mapping, save_name='sigma_scheduler', ylog=True, save=True,
                            **plot_lr_schedule_settings)

    # plot_lr_schedule(sigma_scheduler, training_steps_per_epoch, settings['s_max_epochs'],
    #                  init_learning_rate=settings['s_learning_rate'], save_name='sigma_scheduler',
    #                  y_label='Sigma', title='Sigma scheduler', ylog=False, **plot_lr_schedule_settings)

    plot_images_settings ={
        'ps_runs_path': s_dirs['save_dir'], #'{}/runs'.format(os.getcwd()),
        'ps_run_name': settings['s_sim_name'],
        'ps_device': settings['device'],
        'ps_checkpoint_name': None,  # If none take checkpoint of last epoch
        'ps_inv_normalize': False,
    }

    plot_images_outer(plot_images_settings, **plot_images_settings)

    if settings['s_max_epochs'] > 10:
        plot_images_outer(plot_images_settings, epoch=10, **plot_images_settings)
    # except Exception:
    #     warnings.warn('Image plotting encountered error!')





