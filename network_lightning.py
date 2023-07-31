import torch
import pytorch_lightning as pl

from helper.memory_logging import print_gpu_memory, print_ram_usage
from modules_blocks import Network
import torch.nn as nn
from helper.helper_functions import one_hot_to_mm
from helper.gaussian_smoothing_helper import gaussian_smoothing_target
import torchvision.transforms as T
import copy
import warnings
# Stuff for memory logging


class Network_l(pl.LightningModule):
    def __init__(self, linspace_binning_params, sigma_schedule_mapping, device, s_num_input_time_steps, s_upscale_c_to, s_num_bins_crossentropy,
                 s_width_height, s_learning_rate, s_calculate_quality_params, s_width_height_target, s_max_epochs,
                 s_gaussian_smoothing_target, s_schedule_sigma_smoothing, s_sigma_target_smoothing, s_log_precipitation_difference,
                 training_steps_per_epoch=None, **__):
        super().__init__()
        self.model = Network(c_in=s_num_input_time_steps, s_upscale_c_to=s_upscale_c_to,
                             s_num_bins_crossentropy=s_num_bins_crossentropy, s_width_height_in=s_width_height)

        self.model.to(device)

        self.sigma_schedule_mapping = sigma_schedule_mapping
        self.s_schedule_sigma_smoothing = s_schedule_sigma_smoothing
        self.s_sigma_target_smoothing = s_sigma_target_smoothing

        self.s_learning_rate = s_learning_rate
        self.s_width_height_target = s_width_height_target
        self.s_calculate_quality_params = s_calculate_quality_params
        self.s_max_epochs = s_max_epochs
        self.s_gaussian_smoothing_target = s_gaussian_smoothing_target
        self.s_log_precipitation_difference = s_log_precipitation_difference

        self._linspace_binning_params = linspace_binning_params
        self.training_steps_per_epoch = training_steps_per_epoch
        self.s_device = device

        # Get assigned later
        self.lr_scheduler = None
        self.optimizer = None

        self.train_step_num = 0
        self.val_step_num = 0



    def forward(self, x):
        output = self.model(x)
        return output

    # Uncomment this for no lr_scheduler
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.model.parameters(), lr=self.s_learning_rate)
    #     return optimizer


    def configure_optimizers(self):
        if not self.training_steps_per_epoch is None:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.s_learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 - 3 * 10e-6)
            # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.s_learning_rate,
            #                                                    steps_per_epoch=self.training_steps_per_epoch,
            #                                                    epochs=self.s_max_epochs)
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
            #                                 T_0 = self.training_steps_per_epoch * 10,# Number of iterations for the first restart
            #                                 T_mult = 1, # A factor increases TiTi after a restart
            #                                 eta_min = 1e-5) # Minimum learning rate

            self.optimizer = optimizer
            self.lr_scheduler = copy.deepcopy(lr_scheduler)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    # 'monitor': 'val_loss',  # The metric to monitor for scheduling
                    }
            }

        else:
            warnings.warn('No lr_scheduler as optional training_steps_per_epoch not initialized in Network_l object')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.s_learning_rate)


        return optimizer


    def training_step(self, batch, batch_idx):
        # TODO: DOES THIS BS WORKAROUND WORK??
        self.train_step_num += 1
        input_sequence, target_binned, target, target_one_hot_extended = batch
        # Todo: get rid of float conversion? do this in filter already?

        # TODO: Should this be done in data loading such that workers can distribute compute?
        global_training_step = self.trainer.global_step  # Does this include validation steps??!!!! FCK LIGHTNIng!!!
        #  SEE: https://github.com/Lightning-AI/lightning/discussions/8007


        if self.s_gaussian_smoothing_target:
            if self.s_schedule_sigma_smoothing:
                curr_sigma = self.sigma_schedule_mapping[self.train_step_num]
            else:
                curr_sigma = self.s_sigma_target_smoothing

            target_binned = gaussian_smoothing_target(target_one_hot_extended, device=self.s_device, sigma=curr_sigma,
                                                       kernel_size=128)

        input_sequence = input_sequence.float()
        target_binned = target_binned.float()
        # TODO targets already cropped??
        pred = self.model(input_sequence)

        # loss = nn.KLDivLoss(reduction='batchmean')(pred, target_binned)
        # Reduction= batchmean because:
        # reduction= “mean” (default) doesn’t return the true KL divergence value, please use reduction= “batchmean”
        # which aligns with the mathematical definition.

        # We use cross entropy loss, for both gaussian smoothed and normal one_hot target
        loss = nn.CrossEntropyLoss()(pred, target_binned)

        # self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, sync_dist=True)  # on_step=False, on_epoch=True calculates averages over all steps for each epoch
        # Added sync_dist=True because of:
        # PossibleUserWarning: It is recommended to use `self.log('val_loss', ..., sync_dist=True)`
        # when logging on epoch level in distributed setting to accumulate the metric across devices.

        # MLFlow
        # mlflow.log_metric('train_loss', loss.item())
        ### Additional quality metrics: ###

        if self.s_calculate_quality_params:
            linspace_binning_min, linspace_binning_max, linspace_binning = self._linspace_binning_params
            pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1,
                                    mean_bin_vals=True)
            pred_mm = torch.tensor(pred_mm, device=self.s_device)

        if self.s_calculate_quality_params or self.s_log_precipitation_difference:
            # MSE
            mse_pred_target = torch.nn.MSELoss()(pred_mm, target)
            self.log('train_mse_pred_target', mse_pred_target.item(), on_step=False, on_epoch=True, sync_dist=True)
            # mlflow.log_metric('train_mse_pred_target', mse_pred_target.item())

            # MSE zeros
            mse_zeros_target = torch.nn.MSELoss()(torch.zeros(target.shape, device=self.s_device), target)
            self.log('train_mse_zeros_target', mse_zeros_target, on_step=False, on_epoch=True, sync_dist=True)
            # mlflow.log_metric('train_mse_zeros_target', mse_zeros_target.item())

            persistence = input_sequence[:, -1, :, :]
            persistence = T.CenterCrop(size=self.s_width_height_target)(persistence)
            mse_persistence_target = torch.nn.MSELoss()(persistence, target)
            self.log('train_mse_persistence_target', mse_persistence_target, on_step=False, on_epoch=True, sync_dist=True)
            # mlflow.log_metric('train_mse_persistence_target', mse_persistence_target.item())

        if self.s_log_precipitation_difference:
            with torch.no_grad():
                mean_pred_diff = torch.mean(pred_mm - target).item()
                mean_pred = torch.mean(pred_mm).item()
                mean_target = torch.mean(target).item()

            self.log('train_mean_diff_pred_target_mm', mean_pred_diff, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train_mean_pred_mm', mean_pred, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train_mean_target_mm', mean_target, on_step=False, on_epoch=True, sync_dist=True)

            # self.log('') =

        if self.s_device.type == 'cuda':
            print_gpu_memory()
        print_ram_usage()

        return loss

    def validation_step(self, val_batch, batch_idx):
        # TODO: Does this work??
        self.val_step_num += 1
        input_sequence, target_binned, target, target_one_hot_extended = val_batch

        if self.s_gaussian_smoothing_target:
            if self.s_schedule_sigma_smoothing:
                curr_sigma = self.sigma_schedule_mapping[self.val_step_num]
            else:
                curr_sigma = self.s_sigma_target_smoothing

            target_binned = gaussian_smoothing_target(target_one_hot_extended, device=self.s_device, sigma=curr_sigma,
                                                       kernel_size=128)

        input_sequence = input_sequence.float()
        target_binned = target_binned.float()

        pred = self.model(input_sequence)

        loss = nn.CrossEntropyLoss()(pred, target_binned)

        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)  # , on_step=True

        # mlflow logging without autolog
        # self.logger.experiment.log_metric(self.logger.run_id, 'val_loss', loss.item())

        # mlflow.log_metric('val_loss', loss.item())

        if self.s_calculate_quality_params or self.s_log_precipitation_difference:
            linspace_binning_min, linspace_binning_max, linspace_binning = self._linspace_binning_params
            pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1,
                                    mean_bin_vals=True)
            pred_mm = torch.tensor(pred_mm, device=self.s_device)

        if self.s_calculate_quality_params:
            # MSE
            mse_pred_target = torch.nn.MSELoss()(pred_mm, target)
            self.log('val_mse_pred_target', mse_pred_target.item(), on_step=False, on_epoch=True, sync_dist=True)
            # mlflow.log_metric('val_mse_pred_target', mse_pred_target.item())

            # MSE zeros
            mse_zeros_target = torch.nn.MSELoss()(torch.zeros(target.shape, device=self.s_device), target)
            self.log('val_mse_zeros_target', mse_zeros_target, on_step=False, on_epoch=True, sync_dist=True)
            # mlflow.log_metric('val_mse_zeros_target', mse_zeros_target.item())

            persistence = input_sequence[:, -1, :, :]
            persistence = T.CenterCrop(size=self.s_width_height_target)(persistence)
            mse_persistence_target = torch.nn.MSELoss()(persistence, target)
            self.log('val_mse_persistence_target', mse_persistence_target, on_step=False, on_epoch=True, sync_dist=True)
            # mlflow.log_metric('val_mse_persistence_target', mse_persistence_target.item())

        if self.s_log_precipitation_difference:
            with torch.no_grad():
                mean_pred_diff = torch.mean(pred_mm - target).item()
                mean_pred = torch.mean(pred_mm).item()
                mean_target = torch.mean(target).item()

            self.log('val_mean_diff_pred_target_mm', mean_pred_diff, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_mean_pred_mm', mean_pred, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_mean_target_mm', mean_target, on_step=False, on_epoch=True, sync_dist=True)

        # pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)
        # pred_mm = torch.from_numpy(pred_mm).detach()

        # MSE
        # mse_pred_target = torch.nn.MSELoss()(pred, target)
        # self.log('val_mse_pred_target', mse_pred_target)

        # MSE zeros
        # mse_zeros_target= torch.nn.MSELoss()(torch.zeros(target.shape), target)
        # self.log('val_mse_zeros_target', mse_zeros_target)


