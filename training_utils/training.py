import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler

from baselines import LKBaseline
from helper.helper_functions import save_project_code
from helper.memory_logging import format_duration
from helper.sigma_scheduler_helper import create_scheduler_mapping
from model.logger import create_loggers, TrainingLogsCallback, ValidationLogsCallback, BaselineTrainingLogsCallback, \
    BaselineValidationLogsCallback
from model.model_lightning_wrapper import NetworkL
from training_utils.preprocessing_cache import save_data


def train_wrapper(
        train_data_loader, validation_data_loader,
        training_steps_per_epoch, validation_steps_per_epoch,
        train_time_keys, val_time_keys, test_time_keys,
        train_sample_coords, val_sample_coords,
        radolan_statistics_dict,
        linspace_binning_params,

        settings,
        s_dirs, s_profiling, s_max_epochs, s_sim_name,
        s_gaussian_smoothing_target, s_sigma_target_smoothing, s_schedule_sigma_smoothing,
        s_train_samples_per_epoch, s_val_samples_per_epoch,
        s_calc_baseline,
        s_batch_size,
        s_mode,
        **__
):
    """
    All the junk surrounding train_l() goes in here
    Please keep intput arguments in the same order as the output of create_data_loaders()
    """

    print(f"\nTRAINING DATA LOADER:\n"
          f"  Num samples: {len(train_data_loader.dataset)} "
          f"(Num batches: {len(train_data_loader)})\n"
          f"  Samples per epoch: {s_train_samples_per_epoch if s_train_samples_per_epoch is not None else '= Num samples'}\n"
          f"\nVALIDATION DATA LOADER:\n"
          f"  Num samples: {len(validation_data_loader.dataset)} "
          f"(Num batches: {len(validation_data_loader)})\n"
          f"  Samples per epoch: {s_val_samples_per_epoch if s_val_samples_per_epoch is not None else '= Num samples'}\n")

    print(f"Batch size: {train_data_loader.batch_size}" +
          (f" (different for validation: {validation_data_loader.batch_size})"
           if train_data_loader.batch_size != validation_data_loader.batch_size else ""))

    # Debug information
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    if s_mode == 'cluster':
        # Also print SLURM-specific environment variables
        slurm_vars = [
            "SLURM_JOB_ID", "SLURM_NTASKS", "SLURM_NTASKS_PER_NODE",
            "SLURM_NODELIST", "SLURM_PROCID", "SLURM_LOCALID"
        ]
        for var in slurm_vars:
            print(f"{var}: {os.environ.get(var, 'Not set')}")

    train_logger, val_logger, base_train_logger, base_val_logger = create_loggers(**settings)

    # This is used to save checkpoints of the model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=s_dirs['model_dir'],
        monitor='val_mean_loss',
        filename='model_epoch_{epoch:04d}_valmeanloss_{val_mean_loss:.2f}_best',
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,  # Prevent special characters in file name
    )
    # save_top_k=-1, prevents callback from overwriting previous checkpoints

    if s_profiling:
        profiler = PyTorchProfiler(dirpath=s_dirs['profile_dir'], export_to_chrome=True)
    else:
        profiler = None

    # Increase time out for weights and biases to prevent time out on galavani
    os.environ["WANDB__SERVICE_WAIT"] = "600"

    # Different project names depending on mode:

    wandb_project_name = s_mode


    logger = WandbLogger(name=s_sim_name, project=wandb_project_name)
    # logger = None

    callback_list = [
        checkpoint_callback,
        TrainingLogsCallback(train_logger),
        ValidationLogsCallback(val_logger)
    ]

    if s_gaussian_smoothing_target and s_schedule_sigma_smoothing:

        sigma_schedule_mapping, sigma_scheduler = create_scheduler_mapping(training_steps_per_epoch, s_max_epochs,
                                                                           s_sigma_target_smoothing, **settings)
    else:
        sigma_schedule_mapping, sigma_scheduler = (None, None)

    save_data(
        radolan_statistics_dict,
        train_sample_coords,
        val_sample_coords,
        linspace_binning_params,
        training_steps_per_epoch,
        sigma_schedule_mapping,
        settings,
        s_dirs,
        **__,
    )

    save_project_code(s_dirs['code_dir'])

    print(f"\n STARTING TRAINING \n ...")
    step_start_time = time.time()

    model_l = train_l(
        train_data_loader, validation_data_loader,
        profiler,
        callback_list,
        logger,
        training_steps_per_epoch,
        radolan_statistics_dict,
        linspace_binning_params,
        sigma_schedule_mapping,
        settings,
        **settings)



    print(f'\n DONE. Took {format_duration(time.time() - step_start_time)} \n')

    if s_calc_baseline:
        calc_baselines(**settings,
                       data_loader_list=[train_data_loader, validation_data_loader],
                       logs_callback_list=[BaselineTrainingLogsCallback, BaselineValidationLogsCallback],
                       logger_list=[base_train_logger, base_val_logger],
                       logging_type_list=['train', 'val'],
                       mean_filtered_log_data_list=[radolan_statistics_dict['mean_filtered_log_data'],
                                                radolan_statistics_dict['mean_filtered_log_data']],
                       std_filtered_log_data_list=[radolan_statistics_dict['std_filtered_log_data'],
                                                  radolan_statistics_dict['std_filtered_log_data']],
                       settings=settings
                       )

    # Network_l, training_steps_per_epoch is returned to be able to plot lr_scheduler
    return model_l, training_steps_per_epoch, sigma_schedule_mapping


def train_l(
        train_data_loader, validation_data_loader,
        profiler,
        callback_list,
        logger,
        training_steps_per_epoch,
        radolan_statistics_dict,
        linspace_binning_params,
        sigma_schedule_mapping,

        settings,
        s_max_epochs,
        s_num_gpus,
        s_check_val_every_n_epoch,
        s_validate_on_epoch_0,
        s_save_dir,
        s_mode,
        **__):
    '''
    Train loop
    '''

    # load static and dynamic statistics dicts here from train data loader
    # and pass them to Network_l
    dynamic_statistics_dict_train_data = train_data_loader.dataset.dynamic_statistics_dict
    static_statistics_dict_train_data = train_data_loader.dataset.static_statistics_dict

    # Statistics are only extracted from training data to prevent data leakage and ensure
    # consistency for model learning.

    model_l = NetworkL(
        dynamic_statistics_dict_train_data,
        static_statistics_dict_train_data,
        linspace_binning_params,
        sigma_schedule_mapping,
        settings,
        training_steps_per_epoch=training_steps_per_epoch,
        **settings)

    if s_mode == 'cluster':
        num_gpus = os.environ.get('SLURM_GPUS_ON_NODE')
    else:
        num_gpus = 1

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callback_list,
        profiler=profiler,
        max_epochs=s_max_epochs,
        log_every_n_steps=100,
        logger=logger,
        devices=num_gpus,#'auto',
        check_val_every_n_epoch=s_check_val_every_n_epoch,
        strategy='ddp',
        num_sanity_val_steps=0
    )
    # num_sanity_val_steps=0 turns off validation sanity checking
     # precision='16-mixed'
    # 'devices' argument is ignored when device == 'cpu'
    # Speed up advice: https://pytorch-lightning.readthedocs.io/en/1.8.6/guides/speed.html

    # Optionally perform validation on the untrained model (epoch 0)
    if s_validate_on_epoch_0:
        print("Validating on initialized model (epoch 0)...")
        trainer.validate(model_l, dataloaders=validation_data_loader)
    else:
        print("Skipping validation on initialized model (epoch 0)")
    # trainer.logger = logger
    trainer.fit(model_l, train_data_loader, validation_data_loader)

    # Network_l instance is returned to be able to plot lr_scheduler
    return model_l


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
