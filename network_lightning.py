import torch
from torchvision import transforms
import pytorch_lightning as pl

from helper.calc_CRPS import crps_vectorized
from modules_blocks import Network
from modules_blocks_convnext import ConvNeXtUNet
import torch.nn as nn
from helper.gaussian_smoothing_helper import gaussian_smoothing_target
from helper.sigma_scheduler_helper import bernstein_polynomial, linear_schedule_0_to_1
import torchvision.transforms as T
import copy
import einops
from helper.pre_process_target_input import img_one_hot, inverse_normalize_data, invnorm_linspace_binning, normalize_data
from helper.pre_process_target_input import set_nans_zero, pre_process_target_to_one_hot
from pysteps import verification
from load_data_xarray import random_crop


class NetworkL(pl.LightningModule):
    def __init__(
            self,
            linspace_binning_params,
            sigma_schedule_mapping,
            radolan_statistics_dict,

            settings,
            device,
            s_num_input_time_steps,
            s_num_bins_crossentropy,
            s_width_height_target,
            s_convnext,
            s_crps_loss,
            training_steps_per_epoch=None,
            **__):
        '''
        Both radolan_statistics_dict and  sigma_schedule_mapping  can be None if no training,
        but only forward pass is performed (for checkpoint loading)
        Set training_mode to False for forward pass, when the upper variables are not available during initilization
        '''

        super().__init__()

        self.val_step_num = 0
        self.train_step_num = 0
        # TODO This does not exactly correspond to the number of batches processed:
        #  https://wandb.ai/cognitive_modeling/lightning_logs/runs/5g7hgn02/workspace?nw=nwuserosrip

        # Attributes needed for logging
        self.sum_val_loss = 0
        self.sum_val_loss_squared = 0
        self.sum_val_mse = 0
        self.sum_val_mse_squared = 0
        self.sum_val_mean_pred = 0
        self.sum_val_mean_pred_squared = 0
        self.sum_val_mean_target = 0
        self.sum_val_mean_target_squared = 0

        self.sum_train_loss = 0
        self.sum_train_loss_squared = 0
        self.sum_train_mse = 0
        self.sum_train_mse_squared = 0
        self.sum_train_mean_pred = 0
        self.sum_train_mean_pred_squared = 0
        self.sum_train_mean_target = 0
        self.sum_train_mean_target_squared = 0

        self.sigma_schedule_mapping = sigma_schedule_mapping
        self.settings = settings

        # radolan_statistics_dict can be None if no training, but only forward pass is performed (for checkpoint loading)
        if radolan_statistics_dict is None:
            self.mean_filtered_log_data = None
            self.std_filtered_log_data = None
        else:
            self.mean_filtered_log_data = radolan_statistics_dict['mean_filtered_log_data']
            self.std_filtered_log_data = radolan_statistics_dict['std_filtered_log_data']

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


        # Set up loss function
        # if training_mode:
        if s_crps_loss is None:
            raise ValueError('When using CRPS loss mean_filtered_log_data and std_filtered_log_data have to be passed to Network_l')

        if s_crps_loss:

            linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params
            linspace_binning_inv_norm, linspace_binning_max_inv_norm = invnorm_linspace_binning(
                linspace_binning,
                linspace_binning_max,
                self.mean_filtered_log_data,
                self.std_filtered_log_data)

            self.loss_func = lambda pred, target: torch.mean(crps_vectorized(pred, target,
                                                                  linspace_binning_inv_norm,
                                                                  linspace_binning_max_inv_norm,
                                                                             device))

        else:
            self.loss_func = nn.CrossEntropyLoss()
            # Yields the same result, when inputs are indecies instead of one-hot probabilities for x entropy
            # loss = self.loss_func(pred, torch.argmax(target_binned, dim=1))

        # Initialize model
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

    def forward(self, x):
        output = self.model(x)
        return output

    def configure_optimizers(self):
        s_learning_rate = self.settings['s_learning_rate']
        s_max_epochs = self.settings['s_max_epochs']
        s_lr_schedule = self.settings['s_lr_schedule']

        if s_lr_schedule:
            # Configure optimizer WITH lr_schedule
            if not self.training_steps_per_epoch is None:
                optimizer = torch.optim.AdamW(self.model.parameters(), lr=s_learning_rate)
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                            gamma=1 - 1.5 * (1 / s_max_epochs * (6000 / self.training_steps_per_epoch)) * 10e-4)
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
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=s_learning_rate)
            return optimizer

    def pre_process_target(self, target: torch.Tensor) -> torch.Tensor:
        '''
        Fast on-the fly pre-processing of target in training / validation loop
        Converting into binned / one-hot target
        '''
        s_num_bins_crossentropy = self.settings['s_num_bins_crossentropy']

        # Creating binned target
        linspace_binning_min, linspace_binning_max, linspace_binning = self._linspace_binning_params
        target_binned = img_one_hot(target, s_num_bins_crossentropy, linspace_binning)
        target_binned = einops.rearrange(target_binned, 'b w h c -> b c w h')
        return target_binned

    def pre_process_input(self, input_sequence: torch.Tensor) -> torch.Tensor:
        '''
        Fast on-the-fly pre-processing of input_sequence in training/validation loop
        Replace NaNs with zeros
        '''
        # Replace nans with 0s
        nan_mask = torch.isnan(input_sequence)
        input_sequence[nan_mask] = 0
        return input_sequence

    def dlbd_target_pre_processing(self, extended_target_binned: torch.Tensor) -> torch.Tensor:
        '''
        Fast on-the-fly pre-processing of target in training/validation loop for DLBD
        This requires one-hot target that has been pre-precessed by pre_process_target() method
        The target has to be the larger, extended version that is convolved by the gaussian Kernel
        '''

        s_sigma_target_smoothing = self.settings['s_sigma_target_smoothing']
        s_schedule_sigma_smoothing = self.settings['s_schedule_sigma_smoothing']

        # Getting sigma from scheduler if activated
        if s_schedule_sigma_smoothing:
            curr_sigma = self.sigma_schedule_mapping[self.train_step_num]
        else:
            curr_sigma = s_sigma_target_smoothing

        # Pre-processing target for DLBD
        target_binned = gaussian_smoothing_target(extended_target_binned, device=self.s_device, sigma=curr_sigma,
                                                  kernel_size=128)
        return target_binned

    def train_and_val_step(self, dynamic_samples_dict, static_samples_dict, batch_idx):
        s_gaussian_smoothing_target = self.settings['s_gaussian_smoothing_target']
        s_width_height_target = self.settings['s_width_height_target']
        s_data_variable_name = self.settings['s_data_variable_name']

        # TODO: Is there a better way to handle the dicts here?

        # --- Process Radolan ---
        radolan_spacetime_batch = dynamic_samples_dict['radolan']

        # We start out with a whole unnormalized batched spacetime tensor (b x t x h x w) which has spacial dimensions of the
        # input + augmentation padding and time-wise starts with the first input sequence and ends on the target sequence

        # Augment data
        radolan_spacetime_batch = random_crop(radolan_spacetime_batch, **self.settings)
        # Normalize data
        radolan_spacetime_batch = normalize_data(radolan_spacetime_batch, self.mean_filtered_log_data, self.std_filtered_log_data)
        # Extract target and input
        target = radolan_spacetime_batch[:, -1, :, :]
        radolan_input_sequence = radolan_spacetime_batch[:, 0:4, :, :]

        # Pre-process input and target
        # Convert target into one_hot binned target, all NaNs are assigned zero probabilities for all bins
        target_binned = self.pre_process_target(target)
        # Replace NaNs with Zeros in Input
        radolan_input_sequence = self.pre_process_input(radolan_input_sequence)

        if s_gaussian_smoothing_target:
            target_binned = self.dlbd_target_pre_processing(target_binned)

        # Center crop target to correct size
        center_crop_target = transforms.CenterCrop(s_width_height_target)
        target_binned = center_crop_target(target_binned)
        target = center_crop_target(target)

        radolan_input_sequence = radolan_input_sequence.float()
        target_binned = target_binned.float()

        # --- Process DEM ---
        dem_spatial_batch = static_samples_dict['dem']

        # TODO Augment

        # Normalize
        dem_mean, dem_std = self.trainer.train_dataloader.dataset.static_statistics_dict['dem']
        dem_spatial_batch = (dem_spatial_batch - dem_mean) / dem_std

        pred = self(radolan_input_sequence)

        if torch.isnan(pred).any():
            raise ValueError('NAN in prediction (also leading to nan in loss)')

        loss = self.loss_func(pred, target_binned)

        # returned dict has to include 'loss' entry for automatic backward optimization
        # Multiple entries can be added to the dict, which can be found in 'outputs' of the callback on_train_batch_end()
        # which is currently used by logger.py
        return {
            'loss': loss,  # Loss cannot be NaN
            'pred': pred,  # Pred is inherently not NaN
            'target': target,  # Target can have NaNs (depending on filter condition in pre-processing)
            'target_binned': target_binned  # In target binned for all values that have been NaNs in target simply all bins have been set to zero
        }

    def training_step(self, batched_samples, batch_idx):
        self.train_step_num += 1
        dynamic_samples_dict, static_samples_dict = batched_samples
        out_dict = self.train_and_val_step(dynamic_samples_dict, static_samples_dict, batch_idx)

        return out_dict

    def validation_step(self, batched_samples, batch_idx):
        self.val_step_num += 1
        dynamic_samples_dict, static_samples_dict = batched_samples
        out_dict = self.train_and_val_step(dynamic_samples_dict, static_samples_dict, batch_idx)
        return out_dict
