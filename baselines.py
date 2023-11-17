import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import pysteps.motion as motion
from pysteps import nowcasts
from pysteps import verification
import numpy as np
from load_data import inverse_normalize_data
from helper.helper_functions import one_hot_to_lognorm_mm

class LKBaseline(pl.LightningModule):
    '''
    Optical Flow baseline (PySteps)
    '''
    def __init__(self, logging_type, mean_filtered_data, std_filtered_data, s_num_lead_time_steps, s_calculate_fss,
                 s_fss_scales, s_fss_threshold, device, **__):
        '''
        logging_type depending on data loader either: 'train' or 'val' or None if no logging is desired
        This is used by both,
        '''
        super().__init__()
        self.logging_type = logging_type
        self.mean_filtered_data = mean_filtered_data
        self.std_filtered_data = std_filtered_data

        # Settings
        self.s_num_lead_time_steps = s_num_lead_time_steps
        self.s_calculate_fss = s_calculate_fss
        self.s_fss_scales = s_fss_scales
        self.s_fss_threshold = s_fss_threshold
        self.s_device = device

        self.baseline_method = motion.get_method("LK")


    def forward(self, frames):

        frames_np = frames.detach().cpu().numpy()

        predictions = []
        for batch_idx in range(frames_np.shape[0]):


            # Calculate optical flow using PySTEPS
            # Motion vectors only become feasible for lognormalized data!

            motion_field = self.baseline_method(frames_np[batch_idx, :, :, :])


            # Extrapolate the last radar observation
            extrapolate = nowcasts.get_method("extrapolation")
            precip_forecast = extrapolate(frames_np[batch_idx, -1, :, :], motion_field, self.s_num_lead_time_steps)
            predictions.append(precip_forecast)

        precip_forecast = torch.from_numpy(np.array(predictions)).to(self.s_device)

        return precip_forecast[:, -1, :, :], motion_field, precip_forecast


    def validation_step(self, val_batch, batch_idx):
        input_sequence, target_binned, target, target_one_hot_extended = val_batch
        input_sequence = input_sequence.float()
        target_binned = target_binned.float()

        # TODO: Converting to numpy, as torch inverse normalization does not work
        target_np = target.detach().cpu().numpy()
        target_np = inverse_normalize_data(target_np, self.mean_filtered_data, self.std_filtered_data)
        target = torch.from_numpy(target_np).to(self.s_device)

        pred, _, _ = self(input_sequence)

        pred = T.CenterCrop(size=32)(pred)

        mse_pred_target = torch.nn.MSELoss()(pred, target)

        if self.logging_type is not None:
            self.log('base_{}_mse_pred_target'.format(self.logging_type), mse_pred_target.item(), on_step=False,
                     on_epoch=True, sync_dist=True)



        if self.s_calculate_fss:
            fss = verification.get_method("FSS")
            pred_np = pred.detach().cpu().numpy()
            target_np = target.cpu().numpy()

            for fss_scale in self.s_fss_scales:
                fss_pred_target = np.nanmean(
                    [fss(pred_np[batch_num, :, :], target_np[batch_num, :, :], self.s_fss_threshold, fss_scale)
                     for batch_num in range(np.shape(target_np)[0])]
                )
                if self.logging_type is not None:
                    self.log('base_{}_fss_scale_{:03d}_pred_target'.format(self.logging_type, fss_scale), fss_pred_target,
                             on_step=False, on_epoch=True, sync_dist=True)


class STEPSBaseline(pl.LightningModule):
    '''
    STEPS baseline (PySteps)
    '''
    def __init__(self, logging_type, mean_filtered_data, std_filtered_data, s_num_lead_time_steps, s_calculate_fss,
                 s_fss_scales, s_fss_threshold, device, **__):
        '''
        logging_type depending on data loader either: 'train' or 'val' or None if no logging is desired
        This is used by both,
        '''
        super().__init__()
        self.logging_type = logging_type
        self.mean_filtered_data = mean_filtered_data
        self.std_filtered_data = std_filtered_data

        # Settings
        self.s_num_lead_time_steps = s_num_lead_time_steps
        self.s_calculate_fss = s_calculate_fss
        self.s_fss_scales = s_fss_scales
        self.s_fss_threshold = s_fss_threshold
        self.s_device = device


    def forward(self, frames):
        frames_np = frames.detach().cpu().numpy()
        predictions = []

        for batch_idx in range(frames_np.shape[0]):
            # Generate nowcast using STEPS
            precip_forecast = nowcasts.steps.forecast(frames_np[batch_idx, :, :, :], self.s_num_lead_time_steps, **self.steps_nowcast_kwargs)
            predictions.append(precip_forecast)

        precip_forecast = torch.from_numpy(np.array(predictions)).to(self.s_device)

        return precip_forecast[:, -1, :, :]



    def validation_step(self, val_batch, batch_idx):
        input_sequence, target_binned, target, target_one_hot_extended = val_batch
        input_sequence = input_sequence.float()
        target_binned = target_binned.float()

        # TODO: Converting to numpy, as torch inverse normalization does not work
        target_np = target.detach().cpu().numpy()
        target_np = inverse_normalize_data(target_np, self.mean_filtered_data, self.std_filtered_data)
        target = torch.from_numpy(target_np).to(self.s_device)

        pred, _, _ = self(input_sequence)

        pred = T.CenterCrop(size=32)(pred)

        mse_pred_target = torch.nn.MSELoss()(pred, target)

        if self.logging_type is not None:
            self.log('base_{}_mse_pred_target'.format(self.logging_type), mse_pred_target.item(), on_step=False,
                     on_epoch=True, sync_dist=True)



        if self.s_calculate_fss:
            fss = verification.get_method("FSS")
            pred_np = pred.detach().cpu().numpy()
            target_np = target.cpu().numpy()

            for fss_scale in self.s_fss_scales:
                fss_pred_target = np.nanmean(
                    [fss(pred_np[batch_num, :, :], target_np[batch_num, :, :], self.s_fss_threshold, fss_scale)
                     for batch_num in range(np.shape(target_np)[0])]
                )
                if self.logging_type is not None:
                    self.log('base_{}_fss_scale_{:03d}_pred_target'.format(self.logging_type, fss_scale), fss_pred_target,
                             on_step=False, on_epoch=True, sync_dist=True)