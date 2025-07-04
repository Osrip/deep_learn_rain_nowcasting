import torch
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
import copy
import einops

# Direct imports to avoid cricular dependencies
from helper.calc_CRPS import crps_vectorized
from helper.pre_process_target_input import img_one_hot, invnorm_linspace_binning, normalize_data
from helper.dlbd import dlbd_target_pre_processing

from .conv_next_unet import ConvNeXtUNet


class NetworkL(pl.LightningModule):
    def __init__(
            self,
            dynamic_statistics_dict_train_data,
            static_statistics_dict_train_data,
            linspace_binning_params,
            sigma_schedule_mapping,


            settings,
            device,
            s_num_input_time_steps,
            s_num_bins_crossentropy,
            s_target_height_width,
            s_convnext,
            s_crps_loss,
            training_steps_per_epoch=None,
            **__):

        super().__init__()



        self.mode = 'train' # Please set predict or baseline mode using set_mode() method

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

        # Normlaization statistics
        self.dynamic_statistics_dict_train_data = dynamic_statistics_dict_train_data
        self.static_statistics_dict_train_data = static_statistics_dict_train_data

        # Extract normlaization statistics from radolan as this has to be frequently accessed for logging
        radolan_statistics = self.dynamic_statistics_dict_train_data['radolan']
        self.mean_filtered_log_data = radolan_statistics['mean_filtered_log_data']
        self.std_filtered_log_data = radolan_statistics['std_filtered_log_data']

        self.sigma_schedule_mapping = sigma_schedule_mapping
        self.settings = settings

        self._linspace_binning_params = linspace_binning_params
        self.training_steps_per_epoch = training_steps_per_epoch
        self.s_device = device

        # Get assigned later
        self.lr_scheduler = None
        self.optimizer = None

        self.train_step_num = 0
        self.val_step_num = 0

        # This saves the hyperparameters such that they are loaded by Network_l.load_from_checkpoint() directly
        # without having to reinitialize, see https://github.com/Lightning-AI/pytorch-lightning/issues/4390
        self.save_hyperparameters()


        # Set up loss function
        # if training_mode:
        if s_crps_loss is None:
            raise ValueError('When using CRPS loss mean_filtered_log_data and std_filtered_log_data have to be passed to Network_l')

        if s_crps_loss:

            radolan_statistics = self.dynamic_statistics_dict_train_data['radolan']
            mean_filtered_log_data_radolan = radolan_statistics['mean_filtered_log_data']
            std_filtered_log_data_radolan = radolan_statistics['std_filtered_log_data']

            linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params
            linspace_binning_inv_norm, linspace_binning_max_inv_norm = invnorm_linspace_binning(
                linspace_binning,
                linspace_binning_max,
                mean_filtered_log_data_radolan,
                std_filtered_log_data_radolan
            )

            self.loss_func = lambda pred, target: torch.mean(crps_vectorized(pred, target,
                                                                  linspace_binning_inv_norm,
                                                                  linspace_binning_max_inv_norm,
                                                                             device))

        else:
            self.loss_func = nn.CrossEntropyLoss()
            # TODO potentially include label smoothing (just an arg: label_smoothing = 0.3)
            # Yields the same result, when inputs are indecies instead of one-hot probabilities for x entropy
            # loss = self.loss_func(pred, torch.argmax(target_binned, dim=1))

        # Initialize model
        if s_convnext:
            self.model = ConvNeXtUNet(
                c_list=[32, 64, 128, 256],
                spatial_factor_list=[4, 2, 2],
                num_blocks_list=[1, 2, 4],
                c_in=5,
                c_target=s_num_bins_crossentropy,
                height_width_target=s_target_height_width
            )
        else:
            self.model = Network(c_in=s_num_input_time_steps, **settings)

        self.model.to(device)

    def set_mode(self, mode: str):
        if not mode in ('predict', 'baseline', 'train'):
            raise ValueError('Wrong mode passed to Network_l. Has to be either "predict" or "baseline" or "train"')
        self.mode = mode

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

    def dlbd_method(self, target_binned: torch.Tensor) -> torch.Tensor:
        '''
        Fast on-the-fly pre-processing of target in training/validation loop for DLBD
        This requires one-hot target that has been pre-precessed by pre_process_target() method
        '''

        s_target_height_width = self.settings['s_target_height_width']
        s_sigma_target_smoothing = self.settings['s_sigma_target_smoothing']
        s_schedule_sigma_smoothing = self.settings['s_schedule_sigma_smoothing']

        # Getting sigma from scheduler if activated
        if s_schedule_sigma_smoothing:
            curr_sigma = self.sigma_schedule_mapping[self.train_step_num]
        else:
            curr_sigma = s_sigma_target_smoothing

        # Pre-processing target for DLBD
        # TODO: adjust kernel_size to save compute
        target_binned = dlbd_target_pre_processing(input_tensor=target_binned,
                                                   output_size=s_target_height_width,
                                                   sigma=curr_sigma,
                                                   kernel_size=None)
        sums = target_binned.sum(dim=1)
        # check all close to 1
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6):
            max_dev = (sums - 1).abs().max().item()
            raise ValueError(f"Channel‐sum check failed: max deviation {max_dev:.2e}")

        return target_binned

    def train_val_and_predict_step(self, dynamic_samples_dict, static_samples_dict, batch_idx):
        """
        Training and Validation step

        Data processing:
            dynamic_samples_dict:
                - Data normalization (on training data statistics)
                - Extracting input and target from time space sample
                - pre-process target: --> one hot binning where nans get 0 prob on all bins
                - pre-processing input --> replaces NaNs with Zeros
                - center crop target
            static_samples_dict
                - Data normalization (on all statistics as there is no train/ val / test split here)

            -> Concatenate all samples along channel dim

            - Forward pass
            - Calculate loss from generated prediction

        Input:
            - Samples have been augmented by _get_item_train_() already or are non-padded when received through
             _get_item_predict_()

            dynamic_samples_dict:
                {'variable_name': batched timespace chunk that includes input frames and target frame, torch.Tensor}

                Dictionary, that includes all 'dynamic' variables
                -- thus time-space tensor shape: (batch, time, y | height, x | width)
                len(time) = s_num_input_time_steps + s_num_lead_time_steps + 1

                dynamic_samples_dict['radolan'] receives special treatment, as this is the data that has been filtered

            static_samples_dict:
                {'variable_name': batched spacial chunk, torch.Tensor }
                Dictionary, that includes all 'static' variables
                -- thus space tensor shape: (batch, y | height, x | width)

            General info about all ..._samples_dict:
                - The data is not normalized. All normalization statistics will be calculated
                - The data has already been augmented, thus len(y, x) = s_input_height_width : no padding!

        Output:
            Dictionary with the following           NaNs:
            - 'loss': torch.Tensor (l)              Loss cannot be NaN
            - 'pred': torch.Tensor (b x c x h x w)  Pred is inherently not NaN
            - 'target': torch.Tensor (b x h x w)    Target can have NaNs (depending on filter condition in pre-processing)
            - 'target_binned': torch.Tensor (b x c x h x w),
                                                    In target binned for all values that have been NaNs in target simply
                                                    all bins have been set to zero

            The dictionary can be accessed under the 'outputs' arg in the callbacks
            'loss' is accessed by the trainer to do the backward pass
        """
        s_gaussian_smoothing_target = self.settings['s_gaussian_smoothing_target']
        s_target_height_width = self.settings['s_target_height_width']
        s_data_variable_name = self.settings['s_data_variable_name']

        # --- Process Radolan ---
        radolan_spacetime_batch = dynamic_samples_dict['radolan']

        # We start out with a whole unnormalized batched spacetime tensor (b x t x h x w) which has spacial dimensions
        # of the input + augmentation padding and time-wise starts with the first input sequence and ends on the
        # target sequence

        # Normalize radolan data
        radolan_statistics = self.dynamic_statistics_dict_train_data['radolan']
        mean_filtered_log_data_radolan = radolan_statistics['mean_filtered_log_data']
        std_filtered_log_data_radolan = radolan_statistics['std_filtered_log_data']

        radolan_spacetime_batch = normalize_data(
            radolan_spacetime_batch,
            mean_filtered_log_data_radolan,
            std_filtered_log_data_radolan
        )

        # Extract target and input
        target = radolan_spacetime_batch[:, -1, :, :]
        radolan_input_sequence = radolan_spacetime_batch[:, 0:4, :, :]

        # Pre-process input and target
        # Convert target into one_hot binned target, all NaNs are assigned zero probabilities for all bins
        target_binned = self.pre_process_target(target)
        # Replace NaNs with Zeros in Input
        radolan_input_sequence = self.pre_process_input(radolan_input_sequence)

        center_crop_target = transforms.CenterCrop(s_target_height_width)
        target = center_crop_target(target)

        if s_gaussian_smoothing_target:
            # This reduces H and W to s_target_height_width
            target_binned = self.dlbd_method(target_binned)
        else:
            # Center crop target to correct size
            target_binned = center_crop_target(target_binned)


        radolan_input_sequence = radolan_input_sequence.float()
        target_binned = target_binned.float()

        # --- Process DEM ---
        dem_spatial_batch = static_samples_dict['dem']

        # Normalize
        dem_statistics = self.static_statistics_dict_train_data['dem']
        dem_mean, dem_std = dem_statistics['mean'], dem_statistics['std']
        # Not substracting by mean to keep 0 at 0 and avoid negative values.
        # DEM looks long tail distributed over germany --> potentially normlaize this the same way as rain
        # Normed DEM has max val at 8

        dem_spatial_batch = dem_spatial_batch / dem_std
        # Add channel dim of size 1
        dem_spatial_batch_unsqueezed = dem_spatial_batch.unsqueeze(dim=1)

        # Concatenate along channel dim
        net_input = torch.cat((radolan_input_sequence, dem_spatial_batch_unsqueezed), dim=1)

        # --- Forward Pass ---
        pred = self(net_input)

        if torch.isnan(pred).any():
            raise ValueError('NAN in prediction (also leading to nan in loss)')

        # loss = self.loss_func(pred, target_binned)
        # Yields the same result, when inputs are indecies instead of one-hot probabilities
        loss = self.loss_func(pred, torch.argmax(target_binned, dim=1))

        # returned dict has to include 'loss' entry for automatic backward optimization
        # Multiple entries can be added to the dict, which can be found in 'outputs' of the callback on_train_batch_end()
        # which is currently used by logger.py
        return {
            'loss': loss,  # Loss cannot be NaN
            'pred': pred,  # Pred is inherently not NaN
            'target': target,  # Target can have NaNs (depending on filter condition in pre-processing), and is normalized!
            'target_binned': target_binned  # In target binned for all values that have been NaNs in target simply all bins have been set to zero
        }

    def training_step(self, batched_samples, batch_idx):
        self.train_step_num += 1
        dynamic_samples_dict, static_samples_dict = batched_samples
        out_dict = self.train_val_and_predict_step(dynamic_samples_dict, static_samples_dict, batch_idx)
        return out_dict

    def validation_step(self, batched_samples, batch_idx):
        self.val_step_num += 1
        dynamic_samples_dict, static_samples_dict = batched_samples
        out_dict = self.train_val_and_predict_step(dynamic_samples_dict, static_samples_dict, batch_idx)
        return out_dict

    def predict_step(self, batched_samples, batch_idx: int, dataloader_idx: int = 0):
        '''
        This is called by trainer.predict
        https://lightning.ai/docs/pytorch/stable/common/trainer.html#predict
        '''
        if self.mode == 'predict':
            return self._predict_step_mode_predict_(batched_samples, batch_idx, dataloader_idx)
        elif self.mode == 'baseline':
            return self._predict_step_mode_baseline_(batched_samples, batch_idx, dataloader_idx)
        else:
            raise ValueError('Mode has to be either "predict" or "baseline", when predict_step() is called')

    def _predict_step_mode_predict_(self, batched_samples, batch_idx: int, dataloader_idx: int = 0):
        dynamic_samples_dict, static_samples_dict, sample_metadata_dict = batched_samples
        out_dict = self.train_val_and_predict_step(dynamic_samples_dict, static_samples_dict, batch_idx)
        out_dict['sample_metadata_dict'] = sample_metadata_dict
        return out_dict

    def _predict_step_mode_baseline_(self, batched_samples, batch_idx: int, dataloader_idx: int = 0):
        dynamic_samples_dict, static_samples_dict, baseline = batched_samples

        # Check if DLBD evaluation is enabled in settings
        s_dlbd_eval = self.settings.get('s_dlbd_eval', False)

        if s_dlbd_eval:
            # DLBD evaluation is enabled - use this ugly ass version that adds uncropped target

            # Store the original (uncropped) data for DLBD evaluation
            # First, extract the target from the dynamic samples (before any center cropping)
            radolan_spacetime_batch = dynamic_samples_dict['radolan']

            # Normalize data
            radolan_statistics = self.dynamic_statistics_dict_train_data['radolan']
            mean_filtered_log_data = radolan_statistics['mean_filtered_log_data']
            std_filtered_log_data = radolan_statistics['std_filtered_log_data']

            radolan_spacetime_batch_normalized = normalize_data(
                radolan_spacetime_batch,
                mean_filtered_log_data,
                std_filtered_log_data
            )

            # Save the normalized uncropped target
            target_normalized_uncropped = radolan_spacetime_batch_normalized[:, -1, :, :]

            # Continue with the normal processing
            out_dict = self.train_val_and_predict_step(dynamic_samples_dict, static_samples_dict, batch_idx)

            # Add the baseline and uncropped target to the output dictionary
            out_dict['baseline'] = baseline['baseline']
            # WE NEED THE UNCROPPED TARGET FOR DLBD CALCULATIONS
            out_dict['target_normalized_uncropped'] = target_normalized_uncropped

            return out_dict
        else:
            # DLBD evaluation is disabled - use the original clean implementation
            out_dict = self.train_val_and_predict_step(dynamic_samples_dict, static_samples_dict, batch_idx)
            out_dict['baseline'] = baseline['baseline']
            return out_dict



