import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import pysteps.motion as motion
from pysteps import nowcasts
from pysteps import verification
import numpy as np
from load_data import inverse_normalize_data

class LKBaseline(pl.LightningModule):
    '''
    Optical Flow baseline (PySteps)
    '''
    def __init__(self, logging_type, mean_filtered_log_data, std_filtered_log_data, s_num_lead_time_steps,
                 device, use_steps=False, steps_settings=None, **__):
        '''
        logging_type depending on data loader either: 'train' or 'val' or None if no logging is desired
        This is used by both,
        '''
        super().__init__()
        if use_steps == True:
            self.inference_method = 'steps'
        else:
            self.inference_method = 'extrapolation'

        self.method_calc_motionfield = motion.get_method("LK")

        self.logging_type = logging_type
        self.mean_filtered_log_data = mean_filtered_log_data
        self.std_filtered_log_data = std_filtered_log_data

        # Settings
        self.s_num_lead_time_steps = s_num_lead_time_steps
        self.s_device = device

        self.steps_settings = steps_settings


    def _infer_by_extrapolation(self, frames_2dim, motion_field):
        '''
        This infers the future frame using the motion field implementing extrapolation
        OUt: c (=1) x w x h
        '''
        extrapolate = nowcasts.get_method("extrapolation")
        precip_forecast = extrapolate(frames_2dim, motion_field, self.s_num_lead_time_steps)
        return precip_forecast


    def _infer_with_steps(self, frames_3dim, motion_field, steps_n_ens_members, steps_num_workers, **__):
        '''
        This infers the future frame using the motion field implementing STEPS
        OUt: num_ensembles x c (=1) x w x h
        '''
        STEP = nowcasts.get_method('steps')
        # precip_forecast = STEP(frames_3dim, motion_field, self.s_num_lead_time_steps, mask_method=None)
        precip_forecast = STEP(
            frames_3dim,
            motion_field,
            self.s_num_lead_time_steps,
            n_ens_members=steps_n_ens_members, # num ensemble members TODO: Increase this
            n_cascade_levels=6,
            precip_thr=0.01,  # everything below is assumed to be zero
            kmperpixel=1,
            timestep=5, # Minutes temp. distance between frames of input motion field
            noise_method="nonparametric",
            vel_pert_method="bps",
            mask_method="incremental",
            num_workers=steps_num_workers
            # mask_method=None,
        )
        return precip_forecast


    # Implementation in LDCast Leinonen paper:
    # https: // github.com / MeteoSwiss / ldcast / blob / master / ldcast / models / benchmarks / pysteps.py
    # self.nowcast_method = nowcasts.get_method("steps")

    # Here they seem to be first transforming into mm/h and then into dBz before feeding into
    # STEPS

    # R = self.transform_to_rainrate(x)
    # (R, _) = transformation.dB_transform(
    #     R, threshold=0.1, zerovalue=zerovalue
    # )
    # R[~np.isfinite(R)] = zerovalue
    # if (R == zerovalue).all():
    #     R_f = self.zero_prediction(R, zerovalue)
    # else:
    #     V = dense_lucaskanade(R)
    #     try:
    #         R_f = self.nowcast_method(
    #             R,
    #             V,
    #             self.future_timesteps,
    #             n_ens_members=self.ensemble_size,
    #             n_cascade_levels=6,
    #             precip_thr=threshold,
    #             kmperpixel=self.km_per_pixel,
    #             timestep=self.interval.total_seconds()/60,
    #             noise_method="nonparametric",
    #             vel_pert_method="bps",
    #             mask_method="incremental",
    #             num_workers=2
    #         )
    #         R_f = R_f.transpose(1,2,3,0)

    def forward(self, frames):

        frames_np = frames.detach().cpu().numpy()

        predictions = []
        for batch_idx in range(frames_np.shape[0]):
            # Calculate optical flow using PySTEPS
            # Motion vectors only become feasible for lognormalized data!

            motion_field = self.method_calc_motionfield(frames_np[batch_idx, :, :, :])

            if self.inference_method == 'extrapolation':
                    precip_forecast = self._infer_by_extrapolation(frames_np[batch_idx, -1, :, :], motion_field)
            elif self.inference_method == 'steps':
                if self.steps_settings is None:
                    raise ValueError('When using STEPS, steps settings have to be set with '
                                     'steps_n_ens_members, steps_num_workers')
                precip_forecast = self._infer_with_steps(frames_np[batch_idx, :, :, :], motion_field, **self.steps_settings)
            else:
                raise ValueError(
                    f"Invalid inference method: {self.inference_method}. Expected 'extrapolation' or 'steps'.")

            # Extrapolate the last radar observation

            predictions.append(precip_forecast)

        precip_forecast = torch.from_numpy(np.array(predictions)).to(self.s_device)

        if self.inference_method == 'extrapolation':
            return precip_forecast[:, -1, :, :], motion_field, precip_forecast
        elif self.inference_method == 'steps':
            # steps has one extra dimension for ensemble members
            # Dimension order:
            # [batch, ensemble_member, w, h]
            return precip_forecast[:, :, -1, :, :], motion_field, precip_forecast


    def validation_step(self, val_batch, batch_idx):
        input_sequence, target_binned, target, target_one_hot_extended = val_batch
        input_sequence = input_sequence.float()
        target_binned = target_binned.float()

        # TODO: Converting to numpy, as torch inverse normalization does not work
        target_np = target.detach().cpu().numpy()
        target_np = inverse_normalize_data(target_np, self.mean_filtered_log_data, self.std_filtered_log_data)
        target = torch.from_numpy(target_np).to(self.s_device)

        pred, _, _ = self(input_sequence)

        pred = T.CenterCrop(size=32)(pred)

        mse_pred_target = torch.nn.MSELoss()(pred, target)

        if self.logging_type is not None:
            self.log('base_{}_mse_pred_target'.format(self.logging_type), mse_pred_target.item(), on_step=False,
                     on_epoch=True, sync_dist=True)



