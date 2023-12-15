import torch
import pytorch_lightning as pl

from helper.calc_CRPS import crps_vectorized
from load_data import inverse_normalize_data, invnorm_linspace_binning
from helper.memory_logging import print_gpu_memory, print_ram_usage
from modules_blocks import Network
from modules_blocks_ResNet import ResNet
import torch.nn as nn
from helper.helper_functions import one_hot_to_lognorm_mm
from helper.gaussian_smoothing_helper import gaussian_smoothing_target
from helper.sigma_scheduler_helper import bernstein_polynomial, linear_schedule_0_to_1
import torchvision.transforms as T
import copy
from pysteps import verification
import numpy as np


import warnings
# Stuff for memory logging

class Network_l(pl.LightningModule):
    def __init__(self, linspace_binning_params, sigma_schedule_mapping, data_set_statistics_dict, settings, device, s_num_input_time_steps, s_upscale_c_to, s_num_bins_crossentropy,
                 s_width_height, s_learning_rate, s_calculate_quality_params, s_width_height_target, s_max_epochs,
                 s_gaussian_smoothing_target, s_schedule_sigma_smoothing, s_sigma_target_smoothing, s_log_precipitation_difference,
                 s_lr_schedule, s_calculate_fss, s_fss_scales, s_fss_threshold, s_gaussian_smoothing_multiple_sigmas, s_multiple_sigmas,
                 s_resnet, s_crps_loss, training_steps_per_epoch=None, filter_and_normalization_params=None, **__):
        '''
        Both data_set_statistics_dict and  sigma_schedule_mapping can be None if no training, but only forward pass is performed (for checkpoint loading)
        '''

        super().__init__()
        # self.model = Network(c_in=s_num_input_time_steps, s_upscale_c_to=s_upscale_c_to,
        #                      s_num_bins_crossentropy=s_num_bins_crossentropy, s_width_height_in=s_width_height)

        if s_crps_loss and filter_and_normalization_params is None:
            raise ValueError('When using CRPS loss mean_filtered_data and std_filtered_data have to be passed to Network_l')

        if s_crps_loss:
            # Extract and inverse normalize linspace_binning_params:
            _, mean_filtered_data, std_filtered_data, _, _ = filter_and_normalization_params

            linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params
            linspace_binning_inv_norm, linspace_binning_max_inv_norm = invnorm_linspace_binning(linspace_binning,
                                                                                                linspace_binning_max,
                                                                                                mean_filtered_data,
                                                                                                std_filtered_data)

            self.loss_func = lambda pred, target: torch.mean(crps_vectorized(pred, target,
                                                                  linspace_binning_inv_norm,
                                                                  linspace_binning_max_inv_norm,
                                                                             device))
            # self.loss_func = crps_loss(linspace_binning_inv_norm, linspace_binning_max_inv_norm)
        else:
            self.loss_func = nn.CrossEntropyLoss()

        if not s_resnet:
            self.model = Network(c_in=s_num_input_time_steps, **settings)
        else:
            self.model = ResNet(c_in=s_num_input_time_steps, **settings)

        self.model.to(device)

        self.sigma_schedule_mapping = sigma_schedule_mapping
        self.s_schedule_sigma_smoothing = s_schedule_sigma_smoothing
        self.s_sigma_target_smoothing = s_sigma_target_smoothing

        # data_set_statistics_dict can be None if no training, but only forward pass is performed (for checkpoint loading)
        if data_set_statistics_dict is None:
            self.mean_train_data_set = None
            self.std_train_data_set = None
            self.mean_val_data_set = None
            self.std_val_data_set = None
        else:
            self.mean_train_data_set = data_set_statistics_dict['mean_train_data_set']
            self.std_train_data_set = data_set_statistics_dict['std_train_data_set']
            self.mean_val_data_set = data_set_statistics_dict['mean_val_data_set']
            self.std_val_data_set = data_set_statistics_dict['std_val_data_set']

        self.s_learning_rate = s_learning_rate
        self.s_width_height_target = s_width_height_target
        self.s_calculate_quality_params = s_calculate_quality_params
        self.s_max_epochs = s_max_epochs
        self.s_gaussian_smoothing_target = s_gaussian_smoothing_target
        self.s_log_precipitation_difference = s_log_precipitation_difference
        self.s_lr_schedule = s_lr_schedule
        self.s_calculate_fss = s_calculate_fss
        self.s_fss_scales = s_fss_scales
        self.s_fss_threshold = s_fss_threshold
        self.s_gaussian_smoothing_multiple_sigmas = s_gaussian_smoothing_multiple_sigmas
        self.s_multiple_sigmas = s_multiple_sigmas

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
        if self.s_gaussian_smoothing_multiple_sigmas:
            # chunk output tensor into $num_sigma equal parts along channel dimension, each chunk corresponding to the prediction
            # of one sigma
            output = output.chunk(len(self.s_multiple_sigmas), dim=1)
        return output

    def configure_optimizers(self):
        if self.s_lr_schedule:
            # Configure optimizer WITH lr_schedule
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
                raise ValueError('Optional training_steps_per_epoch not initialized in Network_l object. Cannot proceed without it.')
                # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.s_learning_rate)

        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.s_learning_rate)
            return optimizer


    def training_step(self, batch, batch_idx):
        # TODO: DOES THIS BS WORKAROUND WORK??
        self.train_step_num += 1
        input_sequence, target_binned, target, target_one_hot_extended = batch

        input_sequence = inverse_normalize_data(input_sequence, self.mean_train_data_set, self.std_train_data_set)
        target = inverse_normalize_data(target, self.mean_train_data_set, self.std_train_data_set)

        # Todo: get rid of float conversion? do this in filter already?

        # TODO: Should this be done in data loading such that workers can distribute compute?
        global_training_step = self.trainer.global_step  # Does this include validation steps?
        #  SEE: https://github.com/Lightning-AI/lightning/discussions/8007

        if self.s_gaussian_smoothing_multiple_sigmas:
            smoothed_targets = []
            for sigma in self.s_multiple_sigmas:
                smoothed_targets.append(gaussian_smoothing_target(target_one_hot_extended, device=self.s_device,
                                                          sigma=sigma,
                                                          kernel_size=128).float())

        elif self.s_gaussian_smoothing_target:
            if self.s_schedule_sigma_smoothing:
                curr_sigma = self.sigma_schedule_mapping[self.train_step_num]
            else:
                curr_sigma = self.s_sigma_target_smoothing

            target_binned = gaussian_smoothing_target(target_one_hot_extended, device=self.s_device, sigma=curr_sigma,
                                                       kernel_size=128)



        input_sequence = input_sequence.float()
        target_binned = target_binned.float()
        # TODO targets already cropped??


        # loss = nn.KLDivLoss(reduction='batchmean')(pred, target_binned)
        # Reduction= batchmean because:
        # reduction= “mean” (default) doesn’t return the true KL divergence value, please use reduction= “batchmean”
        # which aligns with the mathematical definition.

        # We use cross entropy loss, for both gaussian smoothed and normal one_hot target
        if not self.s_gaussian_smoothing_multiple_sigmas:
            pred = self(input_sequence)
            loss = self.loss_func(pred, target_binned)

            preds = [pred]
            log_prefixes = ['']
        else:
            log_prefixes = ['sigma_' + str(i) + '_' for i in self.s_multiple_sigmas]

            preds = self(input_sequence)
            losses = []
            for i, (pred_sig, target_binned_sig, log_prefix) in enumerate(zip(preds, smoothed_targets, log_prefixes)):
                # Invert i to start with large value in schedule of the largest sigma
                inverted_i = len(self.s_multiple_sigmas) - i - 1
                if self.s_schedule_sigma_smoothing:
                    loss = self.loss_func(pred_sig, target_binned_sig)
                    x = linear_schedule_0_to_1(self.current_epoch, self.s_max_epochs)
                    weight_curr_sigma = bernstein_polynomial(inverted_i, len(self.s_multiple_sigmas), x)
                    losses.append(loss * weight_curr_sigma)
                else:
                    losses.append(self.loss_func(pred_sig, target_binned_sig))


                linspace_binning_min, linspace_binning_max, linspace_binning = self._linspace_binning_params

                pred_mm = one_hot_to_lognorm_mm(pred_sig, linspace_binning, linspace_binning_max, channel_dim=1,
                                                mean_bin_vals=True)

                # Inverse normalize data
                pred_mm = inverse_normalize_data(pred_mm, self.mean_train_data_set, self.std_train_data_set)

                pred_mm = torch.tensor(pred_mm, device=self.s_device)

                target_mm_sig = one_hot_to_lognorm_mm(target_binned_sig, linspace_binning, linspace_binning_max, channel_dim=1,
                                                      mean_bin_vals=True)
                target_mm_sig = inverse_normalize_data(target_mm_sig, self.mean_train_data_set, self.std_train_data_set)

                target_mm_sig = torch.tensor(target_mm_sig, device=self.s_device)
                # Calculate MSE for each blurred target
                mse_pred_target = torch.nn.MSELoss()(pred_mm, target_mm_sig)
                self.log('train_{}mse_pred_target'.format(log_prefix), mse_pred_target.item(), on_step=False,
                         on_epoch=True, sync_dist=True)

                mse_zeros_target = torch.nn.MSELoss()(torch.zeros(target.shape, device=self.s_device), target_mm_sig)
                self.log('train_{}mse_zeros_target'.format(log_prefix), mse_zeros_target, on_step=False, on_epoch=True, sync_dist=True)

                persistence = input_sequence[:, -1, :, :]
                persistence = T.CenterCrop(size=self.s_width_height_target)(persistence)
                mse_persistence_target = torch.nn.MSELoss()(persistence, target_mm_sig)
                self.log('train_{}mse_persistence_target'.format(log_prefix), mse_persistence_target, on_step=False,
                         on_epoch=True, sync_dist=True)

                # losses = [nn.CrossEntropyLoss()(pred, target_binned) for target_binned in smoothed_targets] #  Old implementation with only one target
            loss = torch.sum(torch.stack(losses))

        log_prefix = ''
        self.log('train_{}loss'.format(log_prefix), loss, on_step=False,
                on_epoch=True, sync_dist=True)  # on_step=False, on_epoch=True calculates averages over all steps for each epoch


        # lognormalisierte preds!!
        if not self.s_gaussian_smoothing_multiple_sigmas:
            for pred, log_prefix in zip(preds, log_prefixes):  # Iterating through predictions and log prefixes for the case of multiple sigmas, which correspinds to multiple predictions

                # self.log('train_loss', loss, on_step=False, on_epoch=True)

                # Added sync_dist=True because of:
                # PossibleUserWarning: It is recommended to use `self.log('val_loss', ..., sync_dist=True)`
                # when logging on epoch level in distributed setting to accumulate the metric across devices.

                # MLFlow
                # mlflow.log_metric('train_loss', loss.item())
                ### Additional quality metrics: ###

                if self.s_calculate_quality_params or self.s_log_precipitation_difference or self.s_calculate_fss:
                    linspace_binning_min, linspace_binning_max, linspace_binning = self._linspace_binning_params
                    pred_mm = one_hot_to_lognorm_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1,
                                                    mean_bin_vals=True)

                    pred_mm = inverse_normalize_data(pred_mm, self.mean_train_data_set, self.std_train_data_set)

                    pred_mm = torch.tensor(pred_mm, device=self.s_device)

                if self.s_calculate_quality_params or self.s_calculate_fss:
                    # MSE
                    mse_pred_target = torch.nn.MSELoss()(pred_mm, target)
                    self.log('train_{}mse_pred_target'.format(log_prefix), mse_pred_target.item(), on_step=False, on_epoch=True, sync_dist=True)
                    # mlflow.log_metric('train_mse_pred_target', mse_pred_target.item())

                    # MSE zeros
                    mse_zeros_target = torch.nn.MSELoss()(torch.zeros(target.shape, device=self.s_device), target)
                    self.log('train_{}mse_zeros_target'.format(log_prefix), mse_zeros_target, on_step=False, on_epoch=True, sync_dist=True)
                    # mlflow.log_metric('train_mse_zeros_target', mse_zeros_target.item())

                    persistence = input_sequence[:, -1, :, :]
                    persistence = T.CenterCrop(size=self.s_width_height_target)(persistence)
                    mse_persistence_target = torch.nn.MSELoss()(persistence, target)
                    self.log('train_{}mse_persistence_target'.format(log_prefix), mse_persistence_target, on_step=False, on_epoch=True, sync_dist=True)
                    # mlflow.log_metric('train_mse_persistence_target', mse_persistence_target.item())

        if self.s_calculate_fss:
            if self.s_gaussian_smoothing_multiple_sigmas:
                # Only calculate FSS for the prediction with the smallest sigma
                pred = preds[np.argmin(self.s_multiple_sigmas)]

            # TODO: UNNORMALIZE THIS BEFORE CALCULATING QUALITY METRICS!!! NOT IMPORTANT FOR MSE BUT FOR FSS (--> Threshold)!!!!!!
            # TODO ALTERRNATIVELY USE LOGNORMALIZED THRESHOLD!!!

            pred_mm = one_hot_to_lognorm_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1,
                                            mean_bin_vals=True)

            pred_mm = inverse_normalize_data(pred_mm, self.mean_train_data_set, self.std_train_data_set)

            pred_mm = torch.tensor(pred_mm, device=self.s_device)

            fss = verification.get_method("FSS")
            target_np = target.detach().cpu().numpy()
            pred_mm_np = pred_mm.detach().cpu().numpy()

            for fss_scale in self.s_fss_scales:

                fss_pred_target = np.nanmean([fss(pred_mm_np[batch_num, :, :], target_np[batch_num, :, :], self.s_fss_threshold, fss_scale)
                                           for batch_num in range(np.shape(target_np)[0])])
                self.log('train_{}fss_scale_{:03d}_pred_target'.format(log_prefix, fss_scale), fss_pred_target, on_step=False, on_epoch=True, sync_dist=True)

                fss_persistence_target = np.nanmean([fss(persistence[batch_num, :, :].detach().cpu().numpy(), target_np[batch_num, :, :], self.s_fss_threshold, fss_scale)
                                                  for batch_num in range(np.shape(target_np)[0])])
                self.log('train_{}fss_scale_{:03d}_persistence_target'.format(log_prefix, fss_scale), fss_persistence_target, on_step=False, on_epoch=True, sync_dist=True)

                fss_zeros_target = np.nanmean([fss(np.zeros(target_np[batch_num, :, :].shape), target_np[batch_num, :, :], self.s_fss_threshold, fss_scale)
                                            for batch_num in range(np.shape(target_np)[0])])
                self.log('train_{}fss_scale_{:03d}_zeros_target'.format(log_prefix, fss_scale), fss_zeros_target, on_step=False, on_epoch=True, sync_dist=True)


        if self.s_log_precipitation_difference:
            with torch.no_grad():
                mean_pred_diff = torch.mean(pred_mm - target).item()
                mean_pred = torch.mean(pred_mm).item()
                mean_target = torch.mean(target).item()

            self.log('train_{}mean_diff_pred_target_mm'.format(log_prefix), mean_pred_diff, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train_{}mean_pred_mm'.format(log_prefix), mean_pred, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train_{}mean_target_mm'.format(log_prefix), mean_target, on_step=False, on_epoch=True, sync_dist=True)

            # self.log('') =


        print_gpu_memory()
        print_ram_usage()

        return loss

    def validation_step(self, val_batch, batch_idx):
        # TODO: Does this work??
        self.val_step_num += 1
        input_sequence, target_binned, target, target_one_hot_extended = val_batch

        input_sequence = inverse_normalize_data(input_sequence, self.mean_val_data_set, self.std_val_data_set)
        target = inverse_normalize_data(target, self.mean_val_data_set, self.std_val_data_set)

        if self.s_gaussian_smoothing_multiple_sigmas:
            smoothed_targets = []
            for sigma in self.s_multiple_sigmas:
                smoothed_targets.append(gaussian_smoothing_target(target_one_hot_extended, device=self.s_device,
                                                          sigma=sigma,
                                                          kernel_size=128).float())

        elif self.s_gaussian_smoothing_target:
            if self.s_schedule_sigma_smoothing:
                curr_sigma = self.sigma_schedule_mapping[self.val_step_num]
            else:
                curr_sigma = self.s_sigma_target_smoothing

            target_binned = gaussian_smoothing_target(target_one_hot_extended, device=self.s_device, sigma=curr_sigma,
                                                       kernel_size=128)


        input_sequence = input_sequence.float()
        target_binned = target_binned.float()


        if not self.s_gaussian_smoothing_multiple_sigmas:
            pred = self(input_sequence)
            loss = self.loss_func(pred, target_binned)

            preds = [pred]
            log_prefixes = ['']
        else:
            log_prefixes = ['sigma_' + str(i) + '_' for i in self.s_multiple_sigmas]

            preds = self(input_sequence)
            losses = []
            for i, (pred_sig, target_binned_sig, log_prefix) in enumerate(zip(preds, smoothed_targets, log_prefixes)):
                # Invert i to start with large value in schedule of the largest sigma
                inverted_i = len(self.s_multiple_sigmas) - i - 1
                if self.s_schedule_sigma_smoothing:
                    loss = self.loss_func(pred_sig, target_binned_sig)
                    x = linear_schedule_0_to_1(self.current_epoch, self.s_max_epochs)
                    weight_curr_sigma = bernstein_polynomial(inverted_i, len(self.s_multiple_sigmas), x)
                    losses.append(loss * weight_curr_sigma)
                else:
                    losses.append(self.loss_func(pred_sig, target_binned_sig))


                linspace_binning_min, linspace_binning_max, linspace_binning = self._linspace_binning_params

                pred_mm = one_hot_to_lognorm_mm(pred_sig, linspace_binning, linspace_binning_max, channel_dim=1,
                                                mean_bin_vals=True)

                # Inverse normalize data
                pred_mm = inverse_normalize_data(pred_mm, self.mean_val_data_set, self.std_val_data_set)

                pred_mm = torch.tensor(pred_mm, device=self.s_device)



                target_mm_sig = one_hot_to_lognorm_mm(target_binned_sig, linspace_binning, linspace_binning_max, channel_dim=1,
                                                      mean_bin_vals=True)
                target_mm_sig = inverse_normalize_data(target_mm_sig, self.mean_val_data_set, self.std_val_data_set)
                target_mm_sig = torch.tensor(target_mm_sig, device=self.s_device)

                # MSE
                mse_pred_target = torch.nn.MSELoss()(pred_mm, target_mm_sig)
                self.log('val_{}mse_pred_target'.format(log_prefix), mse_pred_target.item(), on_step=False,
                         on_epoch=True, sync_dist=True)
                # mlflow.log_metric('val_mse_pred_target', mse_pred_target.item())

                # MSE zeros
                mse_zeros_target = torch.nn.MSELoss()(torch.zeros(target.shape, device=self.s_device), target_mm_sig)
                self.log('val_{}mse_zeros_target'.format(log_prefix), mse_zeros_target, on_step=False, on_epoch=True,
                         sync_dist=True)
                # mlflow.log_metric('val_mse_zeros_target', mse_zeros_target.item())

                persistence = input_sequence[:, -1, :, :]
                persistence = T.CenterCrop(size=self.s_width_height_target)(persistence)
                mse_persistence_target = torch.nn.MSELoss()(persistence, target_mm_sig)
                self.log('val_{}mse_persistence_target'.format(log_prefix), mse_persistence_target, on_step=False,
                         on_epoch=True, sync_dist=True)
                # mlflow.log_metric('val_mse_persistence_target', mse_persistence_target.item())


            # losses = [nn.CrossEntropyLoss()(pred, target_binned) for target_binned in smoothed_targets]
            loss = torch.sum(torch.stack(losses))

        log_prefix = ''

        self.log('val_{}loss'.format(log_prefix), loss, on_step=False, on_epoch=True,
                 sync_dist=True)  # , on_step=True
        # mlflow logging without autolog
        # self.logger.experiment.log_metric(self.logger.run_id, 'val_loss', loss.item())

        # mlflow.log_metric('val_loss', loss.item())

        if not self.s_gaussian_smoothing_multiple_sigmas:
            for pred, log_prefix in zip(preds, log_prefixes):  # Iterating through predictions and log prefixes for the case of multiple sigmas

                if self.s_calculate_quality_params or self.s_log_precipitation_difference or self.s_calculate_fss:
                    linspace_binning_min, linspace_binning_max, linspace_binning = self._linspace_binning_params
                    pred_mm = one_hot_to_lognorm_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1,
                                                    mean_bin_vals=True)
                    pred_mm = inverse_normalize_data(pred_mm, self.mean_val_data_set, self.std_val_data_set)
                    pred_mm = torch.tensor(pred_mm, device=self.s_device)

                if self.s_calculate_quality_params or self.s_calculate_fss:
                    # MSE
                    mse_pred_target = torch.nn.MSELoss()(pred_mm, target)
                    self.log('val_{}mse_pred_target'.format(log_prefix), mse_pred_target.item(), on_step=False, on_epoch=True, sync_dist=True)
                    # mlflow.log_metric('val_mse_pred_target', mse_pred_target.item())

                    # MSE zeros
                    mse_zeros_target = torch.nn.MSELoss()(torch.zeros(target.shape, device=self.s_device), target)
                    self.log('val_{}mse_zeros_target'.format(log_prefix), mse_zeros_target, on_step=False, on_epoch=True, sync_dist=True)
                    # mlflow.log_metric('val_mse_zeros_target', mse_zeros_target.item())

                    persistence = input_sequence[:, -1, :, :]
                    persistence = T.CenterCrop(size=self.s_width_height_target)(persistence)
                    mse_persistence_target = torch.nn.MSELoss()(persistence, target)
                    self.log('val_{}mse_persistence_target'.format(log_prefix), mse_persistence_target, on_step=False, on_epoch=True, sync_dist=True)
                    # mlflow.log_metric('val_mse_persistence_target', mse_persistence_target.item())

        if self.s_calculate_fss:
            if self.s_gaussian_smoothing_multiple_sigmas:
                # Only calculate FSS for the prediction with the smallest sigma
                pred = preds[np.argmin(self.s_multiple_sigmas)]

            pred_mm = one_hot_to_lognorm_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1,
                                            mean_bin_vals=True)
            pred_mm = inverse_normalize_data(pred_mm, self.mean_val_data_set, self.std_val_data_set)
            pred_mm = torch.tensor(pred_mm, device=self.s_device)

            fss = verification.get_method("FSS")
            target_np = target.detach().cpu().numpy()
            pred_mm_np = pred_mm.detach().cpu().numpy()

            for fss_scale in self.s_fss_scales:

                fss_pred_target = np.nanmean([fss(pred_mm_np[batch_num, :, :], target_np[batch_num, :, :], self.s_fss_threshold, fss_scale)
                                           for batch_num in range(np.shape(target_np)[0])])
                self.log('val_{}fss_scale_{:03d}_pred_target'.format(log_prefix, fss_scale), fss_pred_target, on_step=False, on_epoch=True, sync_dist=True)

                fss_persistence_target = np.nanmean([fss(persistence[batch_num, :, :].cpu().numpy(), target_np[batch_num, :, :], self.s_fss_threshold, fss_scale)
                                                  for batch_num in range(np.shape(target_np)[0])])
                self.log('val_{}fss_scale_{:03d}_persistence_target'.format(log_prefix, fss_scale), fss_persistence_target, on_step=False, on_epoch=True, sync_dist=True)

                fss_zeros_target = np.nanmean([fss(np.zeros(target_np[batch_num, :, :].shape), target_np[batch_num, :, :], self.s_fss_threshold, fss_scale)
                                            for batch_num in range(np.shape(target_np)[0])])
                self.log('val_{}fss_scale_{:03d}_zeros_target'.format(log_prefix, fss_scale), fss_zeros_target, on_step=False, on_epoch=True, sync_dist=True)

        if self.s_log_precipitation_difference:
            with torch.no_grad():
                mean_pred_diff = torch.mean(pred_mm - target).item()
                mean_pred = torch.mean(pred_mm).item()
                mean_target = torch.mean(target).item()

            self.log('val_{}mean_diff_pred_target_mm'.format(log_prefix), mean_pred_diff, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_{}mean_pred_mm'.format(log_prefix), mean_pred, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_{}mean_target_mm'.format(log_prefix), mean_target, on_step=False, on_epoch=True, sync_dist=True)

        # pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)
        # pred_mm = torch.from_numpy(pred_mm).detach()

        # MSE
        # mse_pred_target = torch.nn.MSELoss()(pred, target)
        # self.log('val_mse_pred_target', mse_pred_target)

        # MSE zeros
        # mse_zeros_target= torch.nn.MSELoss()(torch.zeros(target.shape), target)
        # self.log('val_mse_zeros_target', mse_zeros_target)


