import torch
import pytorch_lightning as pl

from helper.calc_CRPS import crps_vectorized
from load_data import inverse_normalize_data, invnorm_linspace_binning
from modules_blocks import Network
from modules_blocks_convnext import ConvNeXtUNet
import torch.nn as nn
from helper.helper_functions import one_hot_to_lognorm_mm
from helper.gaussian_smoothing_helper import gaussian_smoothing_target
from helper.sigma_scheduler_helper import bernstein_polynomial, linear_schedule_0_to_1
import torchvision.transforms as T
import copy
from pysteps import verification


class NetworkL(pl.LightningModule):
    def __init__(self, linspace_binning_params, sigma_schedule_mapping, data_set_statistics_dict,
                 settings, device, s_num_input_time_steps, s_num_bins_crossentropy,
                 s_learning_rate, s_width_height_target, s_max_epochs,
                 s_gaussian_smoothing_target, s_schedule_sigma_smoothing, s_sigma_target_smoothing, s_log_precipitation_difference,
                 s_lr_schedule,
                 s_convnext, s_crps_loss,
                 training_steps_per_epoch=None, filter_and_normalization_params=None, class_count_target=None, training_mode=True, **__):
        '''
        Both data_set_statistics_dict and  sigma_schedule_mapping and class_count_target can be None if no training, but only forward pass is
        performed (for checkpoint loading)
        Set training_mode to False for forward pass, when the upper variables are not available during initilization
        '''

        super().__init__()

        # Set up loss function
        # if training_mode:
        if s_crps_loss and filter_and_normalization_params is None:
            raise ValueError('When using CRPS loss mean_filtered_log_data and std_filtered_log_data have to be passed to Network_l')

        if s_crps_loss:
            # Extract and inverse normalize linspace_binning_params:
            _, mean_filtered_log_data, std_filtered_log_data, _, _, _, _ = filter_and_normalization_params

            linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params
            linspace_binning_inv_norm, linspace_binning_max_inv_norm = invnorm_linspace_binning(linspace_binning,
                                                                                                linspace_binning_max,
                                                                                                mean_filtered_log_data,
                                                                                                std_filtered_log_data)

            self.loss_func = lambda pred, target: torch.mean(crps_vectorized(pred, target,
                                                                  linspace_binning_inv_norm,
                                                                  linspace_binning_max_inv_norm,
                                                                             device))

        else:
            self.loss_func = nn.CrossEntropyLoss()

        if s_convnext:
            self.model = ConvNeXtUNet(
                c_list=[4, 32, 64, 128, 256],
                spatial_factor_list=[2, 2, 2, 2],
                num_blocks_list=[2, 2, 2, 2],
                c_target=s_num_bins_crossentropy,
                height_width_target=s_width_height_target
            )
        else:
            self.model = Network(c_in=s_num_input_time_steps, **settings)

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
        self.s_max_epochs = s_max_epochs
        self.s_gaussian_smoothing_target = s_gaussian_smoothing_target
        self.s_log_precipitation_difference = s_log_precipitation_difference


        self.s_lr_schedule = s_lr_schedule

        self._linspace_binning_params = linspace_binning_params
        self.training_steps_per_epoch = training_steps_per_epoch
        self.s_device = device

        # Get assigned later
        self.lr_scheduler = None
        self.optimizer = None

        self.train_step_num = 0
        self.val_step_num = 0

        # This saves the hyperparameters such that they are loaded by Network_l.load_from_checkpoint() directly
        # without having to reinitialze, see https://github.com/Lightning-AI/pytorch-lightning/issues/4390
        self.save_hyperparameters()

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        if self.s_lr_schedule:
            # Configure optimizer WITH lr_schedule
            if not self.training_steps_per_epoch is None:
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.s_learning_rate)
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                            gamma=1 - 1.5 * (1 / self.s_max_epochs * (6000 / self.training_steps_per_epoch)) * 10e-4)
                # https://3.basecamp.com/5660298/buckets/33695235/messages/6386997982

                self.optimizer = optimizer
                self.lr_scheduler = copy.deepcopy(lr_scheduler)

                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': lr_scheduler,
                        }
                }

            else:
                raise ValueError('Optional training_steps_per_epoch not initialized in Network_l object.'
                                 ' Cannot proceed without it.')

        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.s_learning_rate)
            return optimizer

    def training_step(self, batch, batch_idx):
        self.train_step_num += 1
        input_sequence, target_binned, target, target_one_hot_extended = batch

        # Replace nans with 0s
        nan_mask = torch.isnan(input_sequence)
        input_sequence[nan_mask] = 0

        input_sequence = inverse_normalize_data(input_sequence, self.mean_train_data_set, self.std_train_data_set)
        # target = inverse_normalize_data(target, self.mean_train_data_set, self.std_train_data_set)

        if self.s_gaussian_smoothing_target:
            if self.s_schedule_sigma_smoothing:
                curr_sigma = self.sigma_schedule_mapping[self.train_step_num]
            else:
                curr_sigma = self.s_sigma_target_smoothing

            target_binned = gaussian_smoothing_target(target_one_hot_extended, device=self.s_device, sigma=curr_sigma,
                                                       kernel_size=128)

        input_sequence = input_sequence.float()
        target_binned = target_binned.float()

        pred = self(input_sequence)
        if torch.isnan(pred).any():
            raise ValueError('NAN in prediction (also leading to nan in loss)')

        loss = self.loss_func(pred, target_binned)
        # Yields the same result, when inputting indecies instead of one-hot probabilities
        # loss = self.loss_func(pred, torch.argmax(target_binned, dim=1))

        # returned dict has to include 'loss' entry for automatic backward optimization
        # Multiple entries can be added to the dict, which can be found in 'outputs' of the callback on_train_batch_end()
        # which is currently used by logger.py
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        self.val_step_num += 1
        input_sequence, target_binned, target, target_one_hot_extended = val_batch

        input_sequence = inverse_normalize_data(input_sequence, self.mean_val_data_set, self.std_val_data_set)
        target = inverse_normalize_data(target, self.mean_val_data_set, self.std_val_data_set)

        if self.s_gaussian_smoothing_target:
            if self.s_schedule_sigma_smoothing:
                curr_sigma = self.sigma_schedule_mapping[self.val_step_num]
            else:
                curr_sigma = self.s_sigma_target_smoothing

            target_binned = gaussian_smoothing_target(target_one_hot_extended, device=self.s_device, sigma=curr_sigma,
                                                      kernel_size=128)

        input_sequence = input_sequence.float()
        target_binned = target_binned.float()

        pred = self(input_sequence)
        loss = self.loss_func(pred, target_binned)

        return {'loss': loss}
