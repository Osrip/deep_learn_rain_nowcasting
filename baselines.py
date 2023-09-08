import pytorch_lightning as pl
import torch
import pysteps.motion as motion
from pysteps import nowcasts
from helper.helper_functions import one_hot_to_mm

class PyStepsBaseline(pl.LightningModule):
    def __init__(self, s_num_lead_time_steps):
        super().__init__()
        self.s_num_lead_time_steps = s_num_lead_time_steps


    def forward(self, frames, ):
        frames_np = frames.numpy()

        # Calculate optical flow using PySTEPS
        # Motion vectors only become feasible for lognormalized data!
        oflow_method = motion.get_method("LK")
        motion_field = oflow_method(frames_np)

        # Convert the motion field back to a PyTorch tensor
        motion_field = torch.from_numpy(motion_field)

        # Extrapolate the last radar observation
        extrapolate = nowcasts.get_method("extrapolation")
        precip_forecast = extrapolate(frames_np[-1], motion_field, self.s_num_lead_time_steps)

        return precip_forecast[-1, :, :], motion_field, precip_forecast


    def validation_step(self, val_batch):
        input_sequence, target_binned, target, target_one_hot_extended = val_batch

        mse_pred_target = torch.nn.MSELoss()(pred_mm, target)

        self.log('train_mse_pred_target', mse_pred_target.item(), on_step=False, on_epoch=True, sync_dist=True)

#
# class PyStepsBaseline_l(pl.LightningModule):
#     def __init__(self, baseline_model):
#         super().__init__()
#         self.baseline = baseline_model
#
#     def forward(self, x):
#         return self.baseline(x)
