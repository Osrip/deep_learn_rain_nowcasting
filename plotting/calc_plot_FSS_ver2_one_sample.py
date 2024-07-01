import torch
from helper.pre_process_target_input import set_nans_zero, pre_process_target_to_one_hot
from helper.evaluation_metrics import fss
from helper.pre_process_target_input import one_hot_to_lognormed_mm, inverse_normalize_data
import pysteps
from pysteps import verification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from einops import rearrange

def calc_FSS_ver2(
        model,
        data_loader,
        filter_and_normalization_params,
        linspace_binning_params,
        checkpoint_name_no_ending,
        settings,
        ps_device,
        ps_runs_path,
        **__):

    (filtered_indecies,
     mean_filtered_log_data,
     std_filtered_log_data,
     mean_filtered_data,
     std_filtered_data,
     linspace_binning_min_unnormalized,
     linspace_binning_max_unnormalized) = filter_and_normalization_params

    fss_calc_steps = verification.get_method("FSS")

    # Caglars implementation:
    # fss_score =
    # fss(pred,
    # obs, thr=threshold,
    # scale=scale)

    thresholds = np.logspace(-1, 1, 5)
    scales = np.arange(0, 10, 1, dtype=np.int64)
    num_batches: 5

    df_data = []

    with torch.no_grad():
        # Initialize an empty tensor for storing predictions
        predictions = torch.Tensor().to(ps_device)
        targets = torch.Tensor().to(ps_device)

        for i, (input_sequence, target_normed_mm) in enumerate(data_loader):
            input_sequence = input_sequence.to(ps_device)
            target_normed_mm = target_normed_mm.to(ps_device)

            # TODO implement correct pre procesing in other chkpoint plots as well!
            input_sequence = set_nans_zero(input_sequence)
            target_normed_mm = set_nans_zero(target_normed_mm)

            model = model.to(ps_device)
            pred = model(input_sequence)

            # Converting prediction from one-hot to (lognormed) mm
            _, _, linspace_binning = linspace_binning_params
            pred_normed_mm = one_hot_to_lognormed_mm(pred, linspace_binning, channel_dim=1)

            # Inverse normalize target and prediction

            pred_inv_normed_mm = inverse_normalize_data(
                pred_normed_mm,
                mean_filtered_log_data,
                std_filtered_log_data)

            target_inv_normed_mm = inverse_normalize_data(
                target_normed_mm,
                mean_filtered_log_data,
                std_filtered_log_data)

            plt.imshow(target_normed_mm[0, :, :].cpu().numpy())
            plt.title('Target')
            plt.show()

            plt.imshow(pred_inv_normed_mm[0, :, :].cpu().numpy())
            plt.title('Prediction')
            plt.show()

            break

        for threshold in thresholds:

            fss_vals = []

            for scale in scales:

                for batch_idx in range(pred_inv_normed_mm.shape[0]):
                    fss_value = fss_calc_steps(
                        pred_inv_normed_mm[batch_idx, :, :].cpu().numpy(),
                        target_inv_normed_mm[batch_idx, :, :].cpu().numpy(),
                        thr=threshold, scale=scale)
                    fss_vals.append(fss_value)
                    break

            plt.plot(scales, fss_vals)
            plt.title(f'FSS at Threshold {threshold}')
            plt.show()


                    # df_data.append(
                    #     {'fss_mean': fss_value,
                    #      'threshold': threshold,
                    #      'scale': scale}
                    # )

        # plot_df = pd.DataFrame(df_data)





