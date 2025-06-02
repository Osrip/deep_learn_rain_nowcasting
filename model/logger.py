import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger
import torch
from helper.memory_logging import print_gpu_memory, print_ram_usage
from helper.pre_process_target_input import one_hot_to_lognormed_mm, inverse_normalize_data


def logging(prefix_metrics_dict, prefix_train_val, logger, prefix_instance=''):
    """
    prefix_metrics_dict looks like {'mean_loss': mean_loss, 'mean_mse': mean_mse, ...}

    This does all the logging during the training / validation loop
    prefix_train_val has to be either 'train' or 'val'

    Default lightning set-up with this logging:
    Sanity check runs two batches through validation with logging
    Lightning first runs training, then logs training
    then runs validation, then logs validation.
    """

    if prefix_train_val not in ['train', 'val']:
        raise ValueError('prefix_train_val has to be either "train" or "val"')

    with torch.no_grad():
        name_metrics_dict = {}
        for prefix_metric, metric in prefix_metrics_dict.items():
            name_metrics_dict[f'{prefix_train_val}_{prefix_instance}{prefix_metric}'] = metric
        logger.log_metrics(name_metrics_dict)


def create_loggers(s_dirs, **__):
    train_logger = CSVLogger(s_dirs['logs'], name='train_log')
    val_logger = CSVLogger(s_dirs['logs'], name='val_log')

    base_train_logger = CSVLogger(s_dirs['logs'], name='base_train_log')
    base_val_logger = CSVLogger(s_dirs['logs'], name='base_val_log')
    return train_logger, val_logger, base_train_logger, base_val_logger


class TrainingLogsCallback(pl.Callback):
    """
    This inherits and overwrites methods of pytorch_lightning/callbacks/callback.py
    Important info in:
    pytorch_lightning/callbacks/callback.py
    lightning_fabric/loggers/csv_logs.py
    pytorch_lightning/trainer/trainer.py
    """

    def __init__(self, train_logger):
        super().__init__()
        self.logger = train_logger

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, logging_device='cpu'):
        '''
        Make sums of training metrics each batch / iteration
        '''
        # Only log on one GPU in case of multi-gpu run
        if pl_module.device.index == 0:
            if batch_idx == 0:
                pl_module.sum_train_loss = 0
                pl_module.sum_train_loss_squared = 0
                pl_module.sum_train_mse = 0
                pl_module.sum_train_mse_squared = 0
                pl_module.sum_train_mean_pred = 0
                pl_module.sum_train_mean_pred_squared = 0
                pl_module.sum_train_mean_target = 0
                pl_module.sum_train_mean_target_squared = 0
                pl_module.batches_per_epoch_train = 0

            # Unpacking outputs
            loss = outputs['loss']
            pred = outputs['pred']
            target = outputs['target']
            target_binned = outputs['target_binned']

            # Get rid of NaNs in target (set NaN --> 0)
            nan_mask = torch.isnan(target)
            target[nan_mask] = 0

            # Converting prediction from one-hot to (lognormed) mm
            _, _, linspace_binning = pl_module._linspace_binning_params
            pred_normed_mm = one_hot_to_lognormed_mm(pred, linspace_binning, channel_dim=1)

            # Inverse normalize target and prediction

            pred_mm = inverse_normalize_data(pred_normed_mm,
                                             pl_module.mean_filtered_log_data,
                                             pl_module.std_filtered_log_data)

            target = inverse_normalize_data(target,
                                            pl_module.mean_filtered_log_data,
                                            pl_module.std_filtered_log_data)
            # Batch count
            pl_module.batches_per_epoch_train += 1

            # Loss
            pl_module.sum_train_loss += loss
            pl_module.sum_train_loss_squared += loss ** 2

            # (R)MSE
            mse = torch.nn.MSELoss()(pred_mm, target)
            rmse = torch.sqrt(mse)
            pl_module.sum_train_mse += rmse
            pl_module.sum_train_mse_squared += rmse ** 2

            # Mean prediction
            mean_pred = torch.mean(pred_mm)
            pl_module.sum_train_mean_pred += mean_pred
            pl_module.sum_train_mean_pred_squared += mean_pred ** 2

            # Mean target
            mean_target = torch.mean(target)
            pl_module.sum_train_mean_target += mean_target
            pl_module.sum_train_mean_target_squared += mean_target ** 2

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        '''
        Calculate means and stds from the sums of logs each training iteration and log them
        '''
        # Only log on one GPU in case of multi-gpu run
        if pl_module.device.index == 0:
            # Loss
            # Mean
            mean_loss = pl_module.sum_train_loss / pl_module.batches_per_epoch_train
            # Std
            mean_squared_loss = pl_module.sum_train_loss_squared / pl_module.batches_per_epoch_train
            std_loss = (mean_squared_loss - mean_loss ** 2) ** 0.5

            # MSE
            # Mean
            mean_mse = pl_module.sum_train_mse / pl_module.batches_per_epoch_train
            # Std
            mean_squared_mse = pl_module.sum_train_mse_squared / pl_module.batches_per_epoch_train
            std_mse = (mean_squared_mse - mean_mse ** 2) ** 0.5

            # Mean prediction
            # Mean
            mean_mean_pred = pl_module.sum_train_mean_pred / pl_module.batches_per_epoch_train
            # Std
            mean_squared_mean_pred = pl_module.sum_train_mean_pred_squared / pl_module.batches_per_epoch_train
            std_mean_pred = (mean_squared_mean_pred - mean_mean_pred ** 2) ** 0.5

            # Mean target
            # Mean
            mean_mean_target = pl_module.sum_train_mean_target / pl_module.batches_per_epoch_train
            # Std
            mean_squared_mean_target = pl_module.sum_train_mean_target_squared / pl_module.batches_per_epoch_train
            std_mean_target = (mean_squared_mean_target - mean_mean_target ** 2) ** 0.5

            # My CSV logger for plotting pipeline
            self.my_log(
                {'mean_loss': mean_loss, 'std_loss': std_loss,
                 'mean_rmse': mean_mse, 'std_rmse': std_mse,
                 'mean_mean_pred': mean_mean_pred, 'std_mean_pred': std_mean_pred,
                 'mean_mean_target': mean_mean_target, 'std_mean_target': std_mean_target})

            # W and B logger
            pl_module.log_dict(
                {'train_mean_loss': mean_loss,
                 'train_mean_rmse': mean_mse,
                 'train_mean_mean_pred': mean_mean_pred,
                 'train_mean_mean_target': mean_mean_target,
                 'pl_module.batches_per_epoch_train': pl_module.batches_per_epoch_train}
            )

            save_every_n_th_epoch = 1
            if pl_module.current_epoch % save_every_n_th_epoch == 0:
                # Call on_train_end every nth epoch to save intermediate results
                self.on_train_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        # self.train_logger.finalize()
        self.logger.save()

        # Print RAM and VRAM usage
        print_gpu_memory()
        print_ram_usage()

    def my_log(self, prefix_metrics_dict):
        logging(prefix_metrics_dict, 'train', self.logger)


class ValidationLogsCallback(pl.Callback):
    """
    This inherits and overwrites methods of pytorch_lightning/callbacks/callback.py
    Important info in:
    pytorch_lightning/callbacks/callback.py
    lightning_fabric/loggers/csv_logs.py
    pytorch_lightning/trainer/trainer.py
    """

    def __init__(self, val_logger):
        super().__init__()
        self.logger = val_logger

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, logging_device='cpu'):
        '''
        Make sums of training metrics each batch / iteration
        '''
        # Only log on one GPU in case of multi-gpu run
        if pl_module.device.index == 0:
            if batch_idx == 0:
                pl_module.sum_val_loss = 0
                pl_module.sum_val_loss_squared = 0
                pl_module.sum_val_mse = 0
                pl_module.sum_val_mse_squared = 0
                pl_module.sum_val_mean_pred = 0
                pl_module.sum_val_mean_pred_squared = 0
                pl_module.sum_val_mean_target = 0
                pl_module.sum_val_mean_target_squared = 0
                pl_module.batches_per_epoch_val = 0


            # Unpacking outputs
            loss = outputs['loss']
            pred = outputs['pred']
            target = outputs['target']
            target_binned = outputs['target_binned']

            # Get rid of NaNs in target
            nan_mask = torch.isnan(target)
            target[nan_mask] = 0

            # Converting prediction from one-hot to (lognormed) mm
            _, _, linspace_binning = pl_module._linspace_binning_params
            pred_normed_mm = one_hot_to_lognormed_mm(pred, linspace_binning, channel_dim=1)

            # Inverse normalize target and prediction

            pred_mm = inverse_normalize_data(pred_normed_mm,
                                             pl_module.mean_filtered_log_data,
                                             pl_module.std_filtered_log_data)

            target = inverse_normalize_data(target,
                                            pl_module.mean_filtered_log_data,
                                            pl_module.std_filtered_log_data)
            # Batch count
            pl_module.batches_per_epoch_val += 1

            # Loss
            pl_module.sum_val_loss += loss
            pl_module.sum_val_loss_squared += loss ** 2

            # (R)MSE
            mse = torch.nn.MSELoss()(pred_mm, target)
            rmse = torch.sqrt(mse)
            pl_module.sum_val_mse += rmse
            pl_module.sum_val_mse_squared += rmse ** 2

            # Mean prediction
            mean_pred = torch.mean(pred_mm)
            pl_module.sum_val_mean_pred += mean_pred
            pl_module.sum_val_mean_pred_squared += mean_pred ** 2

            # Mean target
            mean_target = torch.mean(target)
            pl_module.sum_val_mean_target += mean_target
            pl_module.sum_val_mean_target_squared += mean_target ** 2

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        '''
        Calculate means and stds from the sums of logs each validation iteration and log them
        '''
        # Only log on one GPU in case of multi-gpu run
        if pl_module.device.index == 0:
            # Loss
            # Mean
            mean_loss = pl_module.sum_val_loss / pl_module.batches_per_epoch_val
            # Std
            mean_squared_loss = pl_module.sum_val_loss_squared / pl_module.batches_per_epoch_val
            std_loss = (mean_squared_loss - mean_loss ** 2) ** 0.5

            # MSE
            # Mean
            mean_mse = pl_module.sum_val_mse / pl_module.batches_per_epoch_val
            # Std
            mean_squared_mse = pl_module.sum_val_mse_squared / pl_module.batches_per_epoch_val
            std_mse = (mean_squared_mse - mean_mse ** 2) ** 0.5

            # Mean prediction
            # Mean
            mean_mean_pred = pl_module.sum_val_mean_pred / pl_module.batches_per_epoch_val
            # Std
            mean_squared_mean_pred = pl_module.sum_val_mean_pred_squared / pl_module.batches_per_epoch_val
            std_mean_pred = (mean_squared_mean_pred - mean_mean_pred ** 2) ** 0.5

            # Mean target
            # Mean
            mean_mean_target = pl_module.sum_val_mean_target / pl_module.batches_per_epoch_val
            # Std
            mean_squared_mean_target = pl_module.sum_val_mean_target_squared / pl_module.batches_per_epoch_val
            std_mean_target = (mean_squared_mean_target - mean_mean_target ** 2) ** 0.5

            # My CSV logger
            self.my_log(
                {'mean_loss': mean_loss, 'std_loss': std_loss,
                 'mean_rmse': mean_mse, 'std_rmse': std_mse,
                 'mean_mean_pred': mean_mean_pred, 'std_mean_pred': std_mean_pred,
                 'mean_mean_target': mean_mean_target, 'std_mean_target': std_mean_target})

            # W and B logger
            pl_module.log_dict(
                {'val_mean_loss': mean_loss,
                 'val_mean_rmse': mean_mse,
                 'val_mean_mean_pred': mean_mean_pred,
                 'val_mean_mean_target': mean_mean_target,
                 'batches_per_epoch_val': pl_module.batches_per_epoch_val}
            )

            save_every_n_th_epoch = 1
            if pl_module.current_epoch % save_every_n_th_epoch == 0:
                # Call on_validation_end every nth epoch to save intermediate results
                self.on_validation_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        # self.train_logger.finalize()
        self.logger.save()

        # Print RAM and VRAM usage
        print_gpu_memory()
        print_ram_usage()

    def my_log(self, prefix_metrics_dict):
        logging(prefix_metrics_dict, 'val', self.logger)


class BaselineTrainingLogsCallback(pl.Callback):
    def __init__(self, base_train_logger):
        super().__init__()
        self.base_train_logger = base_train_logger

    def on_validation_epoch_end(self, trainer, pl_module):
        all_logs = trainer.callback_metrics
        val_logs = {key: value for key, value in all_logs.items() if key.startswith('base_train_')}
        self.base_train_logger.log_metrics(val_logs) # , epoch=trainer.current_epoch)
        self.base_train_logger.save()

    def on_validation_end(self, trainer, pl_module):
        # self.val_logger.finalize()
        self.base_train_logger.save()


class BaselineValidationLogsCallback(pl.Callback):
    def __init__(self, base_val_logger):
        super().__init__()
        self.logger = base_val_logger

    def on_validation_epoch_end(self, trainer, pl_module):
        all_logs = trainer.callback_metrics
        # trainer.callback_metrics = {}
        val_logs = {key: value for key, value in all_logs.items() if key.startswith('base_val_')}
        self.logger.log_metrics(val_logs) # , epoch=trainer.current_epoch)
        # self.val_logger.log_metrics(val_logs) #, step=trainer.current_epoch)
        self.logger.save()

    def on_validation_end(self, trainer, pl_module):
        # self.val_logger.finalize()
        self.logger.save()








