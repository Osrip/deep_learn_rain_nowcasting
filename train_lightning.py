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
    save_tuple_pickle_csv, save_whole_project, load_zipped_pickle, save_data_loader_vars, load_data_loader_vars
import os

import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from logger import ValidationLogsCallback, TrainingLogsCallback, BaselineTrainingLogsCallback, BaselineValidationLogsCallback,\
    create_loggers
from baselines import LKBaseline
import mlflow
from plotting.plotting_pipeline import plotting_pipeline
from helper.sigma_scheduler_helper import create_scheduler_mapping
from helper.helper_functions import no_special_characters
import copy
import warnings


def data_loading(settings, s_force_data_preprocessing, **__):

    if settings['s_log_transform']:
        transform_f = lambda x: torch.log(x + 1) if isinstance(x, torch.Tensor) else np.log(x + 1)
    else:
        transform_f = lambda x: x
    # Try to load data loader vars, if not possible preprocess data
    # If structure of data_loader_vars is changed, change name in _create_save_name_for_data_loader_vars,

    try:
        # When loading data loader vars, the file name is checked for whether log transform was used
        if s_force_data_preprocessing:
            warnings.warn('Forced preprocessing of data as s_force_data_preprocessing == True')
            raise FileNotFoundError('Forced preprocessing of data as s_force_data_preprocessing == True')
        print('Loading data loader vars from file!')
        data_loader_vars = load_data_loader_vars(settings, **settings)
    except FileNotFoundError:
        print('Data loader vars not found, preprocessing data!')
        data_loader_vars = preprocess_data(transform_f, settings, **settings)
        save_data_loader_vars(data_loader_vars, settings, **settings)

    data_set_vars = create_data_loaders(transform_f, *data_loader_vars, settings,  **settings)
    return data_set_vars


def preprocess_data(transform_f, settings, s_ratio_training_data, s_normalize, s_num_bins_crossentropy,
                    s_data_loader_chunk_size, s_linspace_binning_cut_off_unnormalized, **__):


    ###############
    # FILTER, calculation normalization params (mean, std) from filtered targets
    # Save all index chunks that passed filter in filtered_indecies together with normalization statistics and
    # linspace_binning

    (filtered_indecies, mean_filtered_log_data, std_filtered_log_data, mean_filtered_data, std_filtered_data,
     linspace_binning_min_unnormalized, linspace_binning_max_unnormalized) =\
        filtering_data_scraper(transform_f=transform_f, **settings)

    filter_and_normalization_params = (filtered_indecies, mean_filtered_log_data, std_filtered_log_data, mean_filtered_data,
                                       std_filtered_data, linspace_binning_min_unnormalized, linspace_binning_max_unnormalized)

    ############
    # SPLITTING
    # Defining and splitting into training and validation data set
    num_training_samples = int(len(filtered_indecies) * s_ratio_training_data)
    num_validation_samples = len(filtered_indecies) - num_training_samples

    # Randomly split indecies into training and validation indecies, conserving a chunk size of s_data_loader_chunk_size
    # to prevent that validation data can be solved by overfitting / learning by heart
    filtered_indecies_training, filtered_indecies_validation = random_splitting_filtered_indecies(
        filtered_indecies, num_training_samples, s_data_loader_chunk_size)

    ###############
    # LINSPACE BINNING
    # Normalize linspace binning thresholds now that data is available
    linspace_binning_min = lognormalize_data(linspace_binning_min_unnormalized, mean_filtered_log_data, std_filtered_log_data,
                                             transform_f, s_normalize)
    # Subtract a small number to account for rounding errors made in the normalization process
    linspace_binning_max = lognormalize_data(linspace_binning_max_unnormalized, mean_filtered_log_data, std_filtered_log_data,
                                             transform_f, s_normalize)
    # Watch out! mean_filtered_log_data and std_filtered_log_data have been calculated in the log space, as we first take log,
    # then do z normalization!

    # The virtual linspace binning max is used to create the linspace binning, such that the right most bin simply covers all outliers
     # This includes 95% of the data


    # s_linspace_binning_cut_off_unnormalized = 100  # Let's cut that off ad-hoc at 130mm/h, everything obove is sorted into one bin

    linspace_binning_virtual_max = lognormalize_data(s_linspace_binning_cut_off_unnormalized, mean_filtered_log_data,
                                             std_filtered_log_data,
                                             transform_f, s_normalize)

    linspace_binning_min -= 0.001 # Subtract a small number to account for rounding errors made in the normalization process
    linspace_binning_max += 0.001

    # linspace_binning only includes left bin edges. The rightmost bin egde is given by linspace binning max
    linspace_binning = np.linspace(linspace_binning_min, linspace_binning_virtual_max, num=s_num_bins_crossentropy,
                                   endpoint=False)  # num_indecies + 1 as the very last entry will never be used

    ##############
    # WEIGHTING / CLASS FREQUENCIES
    # We calculate the weights of each class which is = `1 / number of samples (pixels)` in class
    # class_weights_target has length and order of bins
    # The class weights don't sum to one --> Why don't I take the softmax of the weights? --> WeightedRandomsampler doesn't care
    class_weights_target, class_count_target, sample_num_target = calc_class_frequencies(filtered_indecies_training, linspace_binning,
                                                                                  mean_filtered_log_data, std_filtered_log_data,
                                                                                  transform_f, settings, normalize=True,
                                                                                  **settings)
    # This calculates the mean weight of each sample, meaning the mean of all pixel weights in the sample are taken
    # target_mean_weights has length and order of targets
    target_mean_weights = class_weights_per_sample(filtered_indecies_training, class_weights_target, linspace_binning,
                                                   mean_filtered_log_data, std_filtered_log_data, transform_f, settings,
                                                   normalize=True)

    print('Size data set: {} \nof which training samples: {}  \nvalidation samples: {}'.format(len(filtered_indecies),
                                                                                                 num_training_samples,
                                                                                                 num_validation_samples))

    return (filtered_indecies_training, filtered_indecies_validation, mean_filtered_log_data, std_filtered_log_data,
            linspace_binning_min, linspace_binning_max, linspace_binning, filter_and_normalization_params,
            target_mean_weights, class_count_target)


def create_data_loaders(transform_f, filtered_indecies_training, filtered_indecies_validation, mean_filtered_log_data, std_filtered_log_data,
                        linspace_binning_min, linspace_binning_max, linspace_binning, filter_and_normalization_params,
                        target_mean_weights, class_count_target, settings, s_batch_size, s_num_workers_data_loader, **__):

    # TODO: RETURN filtered indecies instead of data set
    train_data_set = PrecipitationFilteredDataset(filtered_indecies_training, mean_filtered_log_data, std_filtered_log_data,
                                                  linspace_binning_min, linspace_binning_max, linspace_binning,
                                                  transform_f, **settings)

    validation_data_set = PrecipitationFilteredDataset(filtered_indecies_validation, mean_filtered_log_data,
                                                       std_filtered_log_data,
                                                       linspace_binning_min, linspace_binning_max, linspace_binning,
                                                       transform_f, **settings)

    mean_train_data_set = train_data_set.mean_filtered_log_data
    std_train_data_set = train_data_set.std_filtered_log_data

    mean_val_data_set = validation_data_set.mean_filtered_log_data
    std_val_data_set = validation_data_set.std_filtered_log_data

    data_set_statistics_dict = {'mean_train_data_set': mean_train_data_set,
                                'std_train_data_set': std_train_data_set,
                                'mean_val_data_set': mean_val_data_set,
                                'std_val_data_set': std_val_data_set}

    training_steps_per_epoch = len(train_data_set)

    sampler = WeightedRandomSampler(weights=target_mean_weights, num_samples=training_steps_per_epoch, replacement=True)

    # Does this assume same order in weights as in data_set?? --> Seems so!
    # replacement=True allows for oversampling and in exchange not showing all samples each epoch
    # num_samples gives number of samples per epoch. Setting to len data_set forces sampler to not show all samples each epoch

    train_data_loader = DataLoader(train_data_set, sampler=sampler, batch_size=s_batch_size, drop_last=True,
                                   num_workers=s_num_workers_data_loader, pin_memory=True)

    validation_data_loader = DataLoader(validation_data_set, batch_size=s_batch_size, shuffle=False, drop_last=True,
                                        num_workers=s_num_workers_data_loader, pin_memory=True)


    print('Num training batches: {} \nNum validation Batches: {} \nBatch size: {}'.format(len(train_data_loader),
                                                                                       len(validation_data_loader),
                                                                                       s_batch_size))
    linspace_binning_params = (linspace_binning_min, linspace_binning_max, linspace_binning)
    # tODO: RETURN filtered indecies instead of data set
    return train_data_loader, validation_data_loader, filtered_indecies_training, filtered_indecies_validation,\
        linspace_binning_params, filter_and_normalization_params, training_steps_per_epoch,\
        data_set_statistics_dict, class_count_target
    # training_steps_per_epoch only needed for lr_schedule_plotting


def calc_baselines(data_loader_list, logs_callback_list, logger_list, logging_type_list, mean_filtered_log_data_list,
                   std_filtered_log_data_list, settings, s_epoch_repetitions_baseline, **__):
    '''
    Goes into train_wrapper
    data_loader_list, logs_callback_list, logger_list, logging_type_list have to be in according order
    logging_type depending on data loader either: 'train' or 'val'
    '''


    for data_loader, logs_callback, logger, logging_type, mean_filtered_log_data, std_filtered_log_data in \
            zip(data_loader_list, logs_callback_list, logger_list, logging_type_list, mean_filtered_log_data_list,
                std_filtered_log_data_list):
        # Create callback list in the form of [BaselineTrainingLogsCallback(base_train_logger)]
        callback_list_base = [logs_callback(logger)]


        lk_baseline = LKBaseline(logging_type, mean_filtered_log_data, std_filtered_log_data, **settings)

        trainer = pl.Trainer(callbacks=callback_list_base, max_epochs=s_epoch_repetitions_baseline, log_every_n_steps=1,
                             check_val_every_n_epoch=1)
        trainer.validate(lk_baseline, data_loader)


def train_wrapper(train_data_loader, validation_data_loader, filtered_indecies_training, filtered_indecies_validation,
                  linspace_binning_params, filer_and_normalization_params, training_steps_per_epoch, data_set_statistics_dict,
                  class_count_target, settings, s_dirs, s_model_every_n_epoch, s_profiling, s_max_epochs, s_num_gpus,
                  s_sim_name, s_gaussian_smoothing_target, s_sigma_target_smoothing, s_schedule_sigma_smoothing,
                  s_check_val_every_n_epoch, s_calc_baseline, **__):
    '''
    All the junk surrounding train goes in here
    '''
    # train_data_loader, validation_data_loader, filtered_indecies_training, filtered_indecies_validation, linspace_binning_params, filer_and_normalization_params, training_steps_per_epoch, data_set_statistics_dict, \
    # = data_set_vars


    train_logger, val_logger, base_train_logger, base_val_logger = create_loggers(**settings)

    save_dict_pickle_csv('{}/settings'.format(s_dirs['data_dir']), settings)

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



    save_dict_pickle_csv('{}/data_set_statistcis_dict'.format(s_dirs['data_dir']), data_set_statistics_dict)

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

        sigma_schedule_mapping, sigma_scheduler = create_scheduler_mapping(training_steps_per_epoch, s_max_epochs,
                                                                           s_sigma_target_smoothing, **settings)
    else:
        sigma_schedule_mapping, sigma_scheduler = (None, None)

    model_l = train_l(train_data_loader, validation_data_loader, profiler, callback_list, logger, training_steps_per_epoch,
                      data_set_statistics_dict ,s_max_epochs, linspace_binning_params, s_dirs['data_dir'], s_num_gpus,
                      sigma_schedule_mapping, s_check_val_every_n_epoch, filer_and_normalization_params,
                      class_count_target, settings)

    if s_calc_baseline:
        calc_baselines(**settings,
                       data_loader_list=[train_data_loader, validation_data_loader],
                       logs_callback_list=[BaselineTrainingLogsCallback, BaselineValidationLogsCallback],
                       logger_list=[base_train_logger, base_val_logger],
                       logging_type_list=['train', 'val'],
                       mean_filtered_log_data_list=[data_set_statistics_dict['mean_train_data_set'],
                                                data_set_statistics_dict['mean_val_data_set']],
                       std_filtered_log_data_list=[data_set_statistics_dict['std_train_data_set'],
                                                  data_set_statistics_dict['std_val_data_set']],
                       settings=settings
                       )

    # Save sigma scheduler and training steps per epoch for s_only_plotting
    save_zipped_pickle('{}/training_steps_per_epoch'.format(s_dirs['data_dir']), training_steps_per_epoch)
    save_zipped_pickle('{}/sigma_schedule_mapping'.format(s_dirs['data_dir']), sigma_schedule_mapping)
    # Network_l, training_steps_per_epoch is returned to be able to plot lr_scheduler
    return model_l, training_steps_per_epoch, sigma_schedule_mapping


def train_l(train_data_loader, validation_data_loader, profiler, callback_list, logger, training_steps_per_epoch,
            data_set_statistics_dict, max_epochs, linspace_binning_params, data_dir, num_gpus, sigma_schedule_mapping,
            check_val_every_n_epoch, filter_and_normalization_params, class_count_target, settings):
    '''
    Train loop, keep this clean!
    '''

    model_l = Network_l(linspace_binning_params, sigma_schedule_mapping, data_set_statistics_dict,
                        settings,
                        training_steps_per_epoch=training_steps_per_epoch,
                        filter_and_normalization_params=filter_and_normalization_params,
                        class_count_target=class_count_target,
                        **settings)

    # save_zipped_pickle('{}/Network_l_class'.format(data_dir), model_l)

    if settings['s_resnet']:
        # Used due to bug
        strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'ddp'

    trainer = pl.Trainer(callbacks=callback_list, profiler=profiler, max_epochs=max_epochs, log_every_n_steps=1,
                         logger=logger, devices=num_gpus, check_val_every_n_epoch=check_val_every_n_epoch,
                         strategy=strategy) # on mac: , accelerator='cpu'
    # strategy="ddp", # precision='16-mixed'
    # 'devices' argument is ignored when device == 'cpu'
    # Speed up advice: https://pytorch-lightning.readthedocs.io/en/1.8.6/guides/speed.html

    # trainer.logger = logger
    trainer.fit(model_l, train_data_loader, validation_data_loader)

    # Network_l is returned to be able to plot lr_scheduler
    return model_l


def create_s_dirs(sim_name, s_local_machine_mode):

    s_dirs = {}
    if s_local_machine_mode:
        s_dirs['save_dir'] = 'runs/{}'.format(sim_name)
    else:
        s_dirs['save_dir'] = '/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/runs/{}'.format(sim_name)

    # s_dirs['save_dir'] = 'runs/{}'.format(s_sim_name)
    s_dirs['plot_dir'] = '{}/plots'.format(s_dirs['save_dir'])
    s_dirs['plot_dir_images'] = '{}/images'.format(s_dirs['plot_dir'])
    s_dirs['plot_dir_fss'] = '{}/fss'.format(s_dirs['plot_dir'])
    s_dirs['model_dir'] = '{}/model'.format(s_dirs['save_dir'])
    s_dirs['code_dir'] = '{}/code'.format(s_dirs['save_dir'])
    s_dirs['profile_dir'] = '{}/profile'.format(s_dirs['save_dir'])
    s_dirs['logs'] = '{}/logs'.format(s_dirs['save_dir'])
    s_dirs['data_dir'] = '{}/data'.format(s_dirs['save_dir'])

    return s_dirs


if __name__ == '__main__':


    #  Training data
    # num_training_samples = 20  # 1000  # Number of loaded pictures (first pics not used for training but only input)
    # num_validation_samples = 20  # 600

    # train_start_date_time = datetime.datetime(2020, 12, 1)
    # s_folder_path = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/weather_data/dwd_datensatz_bits/rv_recalc/RV_RECALC/hdf/'

    s_local_machine_mode = False

    s_force_data_preprocessing = True  # This forces data preprocessing instead of attempting to load preprocessed data

    s_sim_name_suffix = 'default_64_bins_100mm_h_cut_off_fixed_crop_index_bug_ALSO_TARGET_corr_snapshot_7_epochs'  # 'bernstein_scheduler_0_1_0_5_1_2' #'no_gaussian_blurring__run_3_with_lt_schedule_100_epoch_eval_inv_normalized_eval' # 'No_Gaussian_blurring_with_lr_schedule_64_bins' #'sigma_init_5_exp_sigma_schedule_WITH_lr_schedule_xentropy_loss_20_min_lead_time'#'scheduled_sigma_exp_init_50_no_lr_schedule_100G_mem' #'sigma_50_no_sigma_schedule_no_lr_schedule' #'scheduled_sigma_exp_init_50_no_lr_schedule_100G_mem'# 'sigma_50_no_sigma_schedule_lr_init_0_001' # 'scheduled_sigma_exp_init_50_lr_init_0_001' #'no_gaussian_smoothing_lr_init_0_001' #'' #'scheduled_sigma_exp_init_50_lr_init_0_001' #'no_gaussian_smoothing_lr_init_0_001' #'scheduled_sigma_cos_init_20_to_0_1_lr_init_0_001' #'smoothing_constant_sigma_1_and_lr_schedule' #'scheduled_sigma_cos_init_20_to_0_1_lr_init_0_001'
    # _1_2_4_
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

    s_dirs = create_s_dirs(s_sim_name, s_local_machine_mode)

    settings = \
        {
            's_local_machine_mode': s_local_machine_mode,
            's_force_data_preprocessing': s_force_data_preprocessing,
            's_sim_name': s_sim_name,
            's_sim_same_suffix': s_sim_name_suffix,

            's_resnet': False,  # Use ResNet instead of ours

            # TODO: Implement!!
            's_plotting_only': False,  # If active loads sim s_plot_sim_name and runs plotting pipeline
            's_plot_sim_name': 'Run_20240126-224535_ID_51437Weighted_x_entropy_loss', #_2_4_8_16_with_plotting_fixed_plotting', #'Run_20231005-144022TEST_several_sigmas_2_4_8_16_with_plotting_fixed_plotting',
            's_save_prefix_data_loader_vars': 's_save_prefix_data_loader_vars_2_std_linspace_binning',

            's_max_epochs': 7,  #10  # default: 50 Max number of epochs, affects scheduler (if None: runs infinitely, does not work with scheduler)
            's_folder_path': '/mnt/qb/work2/butz1/bst981/weather_data/dwd_nc/zarr',  #'/mnt/qb/work2/butz1/bst981/weather_data/benchmark_data_set',
            's_data_file_name': 'RV_recalc.zarr',  #'yw_done.zarr',
            's_data_variable_name': 'RV_recalc',
            's_data_preprocessing_chunk_num': 50, #Number of chunks that are loaded into ram during pre-processing --> Number of chunks that the above data set is split into 2 year radolan zarr dataset is about 1 tb as np array
            's_choose_time_span': False,
            's_time_span': (datetime.datetime(2020, 12, 1), datetime.datetime(2020, 12, 1)),
            's_ratio_training_data': 0.6,
            's_data_loader_chunk_size': 288,  # 20, #  Chunk size, that consecutive data is chunked in when performing random splitting
            # Changed on 9.2.24 from 20 to 288 (corresponds to 24h)
            's_num_workers_data_loader': 8, # Should correspond to number of cpus, also increases cpu ram
            's_check_val_every_n_epoch': 1, # Calculate validation every nth epoch for speed up, NOT SURE WHETHER PLOTTING CAN DEAL WITH THIS BEING LARGER THAN 1 !!

            # Parameters related to lightning
            's_num_gpus': 4,
            's_batch_size': 64, #48, # 2080--> 18 läuft 2080-->14 --> 7GB /10GB; v100 --> 45  55; a100 --> 64, downgraded to 45 after memory issue on v100 with smoothing stuff
            # resnet 34 original res blocks on a100 --> batch size 32 (tested 64, which did not work)
            # Make this divisible by 8 or best 8 * 2^n

            # Parameters that give the network architecture
            's_upscale_c_to': 32,  # 64, #128, # 512,
            's_num_bins_crossentropy': 64,  # 64, #256,

            # Parameters that give binning
            's_linspace_binning_cut_off_unnormalized': 100,
            # Let's cut that off ad-hoc (in mm/h) , everything obove is sorted into the last bin

            # 'minutes_per_iteration': 5,
            's_width_height': 256,
            's_width_height_target': 32,
            's_num_samples_per_frame': 20,  # Number of samples that are generated from one frame in the data set
            's_num_epochs': 1000,  # TODO: What does this do? Seems to be legacy and not in use anymore
            's_num_input_time_steps': 4,  # The number of subsequent time steps that are used for one predicition
            's_num_lead_time_steps': 3, # 0 --> 0 min prediction (target == last input) ; 1 --> 5 min predicition, 3 --> 15min etc
            # This is substracted by 2: settings['s_num_lead_time_steps'] = 's_num_lead_time_steps' -2 for following reasons:
            # 5, # The number of pictures that are skipped from last input time step to target, starts with -1
            # (starts counting at filtered_data_loader_indecies_dict['last_idx_input_sequence'], where last index is excess
            # for arange ((np.arange(1:5) = [1,2,3,4])
            's_optical_flow_input': False,  # Not yet working!
            # batch size 22: Total: 32G, Free: 6G, Used:25G | Batch size 26: Total: 32G, Free: 1G, Used:30G --> vielfache von 8 am besten
            's_save_trained_model': True,  # saves model every epoch
            's_load_model': False,
            's_load_model_name': 'Run_·20230220-191041',
            's_dirs': s_dirs,
            'device': device,
            's_learning_rate': 0.001,  # 0.0001
            # For some reason the lr scheduler starts one order of magitude below the given learning rate (10^-4, when 10^-3 is given)
            's_lr_schedule': True,  # enables lr scheduler, takes s_learning_rate as initial rate

            # Loss
            's_crps_loss': False,  # CRPS loss instead of X-entropy loss
            's_weighted_loss': False,  #Loss weighted acc. to inverse of class/bin frequency, ONLY WORKS FOR XENTROPY loss

            # DLBD, Gaussian smoothing
            's_gaussian_smoothing_target': False,
            's_sigma_target_smoothing': 0.1,  # In case of scheduling this is the initial sigma
            's_schedule_sigma_smoothing': False,
            's_gaussian_smoothing_multiple_sigmas': False, # ignores s_gaussian_smoothing_target, s_sigma_target_smoothing and s_schedule_sigma_smoothing, s_schedule_multiple_sigmas activates scheduling for multiple sigmas
            's_multiple_sigmas': [0.1, 0.5, 1, 2], # FOR SCHEDULING MAKE SURE LARGEST SIGMA IS LAST, List of sigmas in case s_gaussian_smoothing_multiple_sigmas == True; to create loss mean is taken of all losses that each single sigma would reate
            # ! left most sigma prediction is the one that is plotted. Usually this is close to zero such that it is almost pixel-wise!
            's_schedule_multiple_sigmas': False, # Bernstein scheduling: Schedule multiple sigmas with bernstein polynomial,

            # Logging
            's_calc_baseline': True,  # Baselines are calculated and plotted --> Optical flow baseline
            's_epoch_repetitions_baseline': 1000 ,  # Number of repetitions of baseline calculation; average is taken; each epoch is done on one batch by dataloader
            's_log_precipitation_difference': True,
            's_calculate_quality_params': True,  # Calculating quality params during training and validation
            's_calculate_fss': True,  # Calculating fractions skill score during training and validation
            's_fss_scales': [2, 4, 8, 12],  #[2, 16, 32], # Scales for which fss is calculated as a list
            's_fss_threshold': 1,  # Threshold in mm/h for which fss is calculated

            # Log transform input/ validation data --> log binning --> log(x+1)
            's_log_transform': True,  # False not tested, leave this true
            's_normalize': True,  # False not tested, leave this true

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

            # Save data loader variables
            's_data_loader_vars_path': '/mnt/qb/work2/butz1/bst981/weather_data/data_loader_vars',
            # Max number of frames in proccessed data set for debugging (validation + training)
            's_max_num_filter_hits': None,  # [Disabled when set to None]
        }

    if not settings['s_plotting_only']:
        for _, make_dir in s_dirs.items():
            if not os.path.exists(make_dir):
                os.makedirs(make_dir)

    if settings['s_no_plotting']:
        for en in ['s_plot_average_preds_boo', 's_plot_pixelwise_preds_boo', 's_plot_target_vs_pred_boo',
                   's_plot_mse_boo', 's_plot_losses_boo', 's_plot_img_histogram_boo']:
            settings[en] = False

    if settings['s_local_machine_mode']:

        settings['s_data_variable_name'] = 'RV_recalc'
        # settings['s_data_variable_name'] = 'yw_radolan'

        settings['s_folder_path'] = 'dwd_nc/own_test_data'
        # settings['s_folder_path'] = 'dwd_nc/benchmark_set_test_data'

        # settings['s_data_file_names'] = ['RV_recalc_data_2019-01_subset.nc']
        # settings['s_data_file_name'] = 'yw_2019_6_11.zarr'
        settings['s_data_file_name'] = 'testdata_two_days_2019_01_01-02.zarr'

        settings['s_data_preprocessing_chunk_num'] = 2

        # settings['s_choose_time_span'] = True
        settings['s_choose_time_span'] = False  # DO NOT USE, screws up indecies for data loading
        # settings['s_time_span'] = (datetime.datetime(2019, 1, 1, 0), datetime.datetime(2019, 1, 1, 5))
        settings['s_time_span'] = (67, 150)  # Only used when s_choose_time_span == True; now done according to index (isel instead of sel)
        settings['s_upscale_c_to'] = 32  # 8
        settings['s_batch_size'] = 2
        settings['s_data_loader_chunk_size'] = 1
        settings['s_testing'] = True  # Runs tests at the beginning
        settings['s_min_rain_ratio_target'] = 0  # Deactivated # No Filter
        settings['s_num_workers_data_loader'] = 0  # Debugging only works with zero workers
        settings['s_max_epochs'] = 1  # 3
        settings['s_num_gpus'] = 1

        settings['s_multiple_sigmas'] = [2, 16]
        settings['s_data_loader_vars_path'] = '/home/jan/Programming/weather_data/data_loader_vars' #'/mnt/qb/work2/butz1/bst981/weather_data/data_loader_vars' #
        settings['s_max_num_filter_hits'] = 4
        # FILTER NOT WORKING YET, ALWAYS RETURNS TRUE FOR TEST PURPOSES!!

    # mlflow.create_experiment(settings['s_sim_name'])
    # mlflow.set_tag("mlflow.runName", settings['s_sim_name'])
    # mlflow.pytorch.autolog()
    # mlflow.log_models = False

    if not settings['s_plotting_only']:
        # Normal training
        data_set_vars = data_loading(settings, **settings)
        model_l, training_steps_per_epoch, sigma_schedule_mapping = train_wrapper(*data_set_vars, settings,
                                                                                  **settings)
        plotting_pipeline(sigma_schedule_mapping, training_steps_per_epoch, model_l, settings, **settings)

    else:
        # Plotting only
        load_dirs = create_s_dirs(settings['s_plot_sim_name'], settings['s_local_machine_mode'])
        training_steps_per_epoch = load_zipped_pickle('{}/training_steps_per_epoch'.format(load_dirs['data_dir']))
        sigma_schedule_mapping = load_zipped_pickle('{}/sigma_schedule_mapping'.format(load_dirs['data_dir']))
        settings_loaded = load_zipped_pickle('{}/settings'.format(load_dirs['data_dir']))
        # Convert some of the loaded settings to the current settings
        settings_loaded['s_num_gpus'] = settings['s_num_gpus']

        plotting_pipeline(sigma_schedule_mapping, training_steps_per_epoch, model_l=None, s_dirs=load_dirs,
                          settings=settings_loaded, plot_lr_schedule_boo=False)









