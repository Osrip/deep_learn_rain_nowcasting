import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import pysteps.motion as motion
from pysteps import nowcasts
from pysteps import verification
import numpy as np
from helper.helper_functions import one_hot_to_mm

class LKBaseline(pl.LightningModule):
    '''
    Optical Flow baseline (PySteps)
    '''
    def __init__(self, logging_type, s_num_lead_time_steps, s_calculate_fss, s_fss_scales, s_fss_threshold, device, **__):
        '''
        logging_type depending on data loader either: 'train' or 'val'
        '''
        super().__init__()
        self.logging_type = logging_type
        self.s_num_lead_time_steps = s_num_lead_time_steps
        self.s_calculate_fss = s_calculate_fss
        self.s_fss_scales = s_fss_scales
        self.s_fss_threshold = s_fss_threshold
        self.s_device = device


    def forward(self, frames):

        frames_np = frames.detach().cpu().numpy()

        predictions = []
        for batch_idx in range(frames_np.shape[0]):


            # Calculate optical flow using PySTEPS
            # Motion vectors only become feasible for lognormalized data!
            oflow_method = motion.get_method("LK")
            motion_field = oflow_method(frames_np[batch_idx, :, :, :])


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

        pred, _, _ = self(input_sequence)

        pred = T.CenterCrop(size=32)(pred)

        mse_pred_target = torch.nn.MSELoss()(pred, target)

        self.log('base_{}_mse_pred_target'.format(self.logging_type), mse_pred_target.item(), on_step=False,
                 on_epoch=True, sync_dist=True)



        if self.s_calculate_fss:
            fss = verification.get_method("FSS")
            pred_np = pred.detach().cpu().numpy()
            target_np = target.cpu().numpy()

            for fss_scale in self.s_fss_scales:
                fss_pred_target = np.mean(
                    [fss(pred_np[batch_num, :, :], target_np[batch_num, :, :], self.s_fss_threshold, fss_scale)
                     for batch_num in range(np.shape(target_np)[0])]
                )
                self.log('base_{}_fss_scale_{:03d}_pred_target'.format(self.logging_type, fss_scale), fss_pred_target, on_step=False, on_epoch=True,
                         sync_dist=True)

#
# class PyStepsBaseline_l(pl.LightningModule):
#     def __init__(self, baseline_model):
#         super().__init__()
#         self.baseline = baseline_model
#
#     def forward(self, x):
#         return self.baseline(x)
