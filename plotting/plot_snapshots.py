import numpy as np
import torch

from helper.helper_functions import one_hot_to_lognorm_mm, convert_to_binning_and_back
from load_data import inverse_normalize_data, lognormalize_data, invnorm_linspace_binning
from plotting.plot_images import plot_target_vs_pred_with_likelihood
from plotting.plot_distributions_in_snapshot import plot_distributions
from baselines import LKBaseline
import torchvision.transforms as T



def plot_snapshots(model, data_loader, filter_and_normalization_params, linspace_binning_params, transform_f, settings, plot_settings,
                   ps_runs_path, ps_run_name, ps_checkpoint_name, ps_device, ps_inv_normalize,
                   ps_gaussian_smoothing_multiple_sigmas, ps_multiple_sigmas, prefix='', plot_baseline=True, **__):

    with (torch.no_grad()):

        filtered_indecies, mean_filtered_log_data, std_filtered_log_data, _, _, linspace_binning_min_unnormalized,\
            linspace_binning_max_unnormalized = filter_and_normalization_params

        linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params

        linspace_binning_inv_norm, linspace_binning_max_inv_norm = invnorm_linspace_binning(linspace_binning,
                                                                                            linspace_binning_max,
                                                                                            mean_filtered_log_data,
                                                                                            std_filtered_log_data)

        inv_norm = lambda x: inverse_normalize_data(x, mean_filtered_log_data, std_filtered_log_data, inverse_log=True,
                                                               inverse_normalize=True)
        if ps_inv_normalize:
            inv_norm_or_not = inv_norm
        else:
            inv_norm_or_not = lambda x: x

        for i, (input_sequence, target_one_hot, target, _) in enumerate(data_loader):
            input_sequence = input_sequence.to(ps_device)

            model = model.to(ps_device)
            pred = model(input_sequence)

            if plot_baseline:
                logging_type = None
                lk_baseline = LKBaseline(logging_type, mean_filtered_log_data, std_filtered_log_data, **settings)
                # Optical flow untly works in unnormalized space (no negative values), therefore always inv normalize
                input_sequence_inv_normed = inv_norm(input_sequence).to('cpu')
                pred_mm_baseline, _, _ = lk_baseline(input_sequence_inv_normed)
                pred_mm_baseline = T.CenterCrop(size=32)(pred_mm_baseline)
                pred_mm_baseline = pred_mm_baseline.detach().cpu().numpy()
                # renormalize in case the rest of the plots are not inv normalized
                if not ps_inv_normalize:
                    pred_mm_baseline = lognormalize_data(pred_mm_baseline, mean_filtered_log_data, std_filtered_log_data,
                                                         transform_f, settings['s_normalize'])

                # pred_mm_baseline = convert_to_binning_and_back(pred_mm_baseline, linspace_binning, linspace_binning_max)

            # target = convert_to_binning_and_back(target.detach().cpu().numpy(), linspace_binning, linspace_binning_max)
            # target = torch.from_numpy(target).to(ps_device)

            if not ps_gaussian_smoothing_multiple_sigmas:
                preds = [pred]
                sigma_strs = ['']
            else:
                preds = pred
                sigma_strs = ['_sigma_{}'.format(sigma) for sigma in ps_multiple_sigmas]

            # When s_gaussian_smoothing_multiple_sigmas we get several predictions, which we iterate through
            for pred, sigma_str in zip(preds, sigma_strs):


                pred_mm = one_hot_to_lognorm_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)

                # min and max for the snapshots. Max is 4 x std. Masking Nans
                nan_mask_target = torch.isnan(target)
                nan_mask_input_seq = torch.isnan(input_sequence)
                vmin = min(torch.min(inv_norm_or_not(target[~nan_mask_target])).item(),
                           torch.min(inv_norm_or_not(input_sequence[~nan_mask_input_seq])).item())

                vmax = float(torch.mean(inv_norm_or_not(input_sequence[~nan_mask_input_seq]))
                + 4 * torch.std(inv_norm_or_not(input_sequence[~nan_mask_input_seq])))

                if i == 0:
                    # !!! Can also be plotted without input sequence by just leaving input_sequence=None !!!
                    plot_target_vs_pred_with_likelihood(inv_norm_or_not(target), inv_norm_or_not(pred_mm), pred, pred_mm_baseline,
                                                        plot_baseline=plot_baseline,
                                                        linspace_binning=inv_norm_or_not(linspace_binning),
                                                        vmin=vmin,
                                                        vmax=vmax,
                                                        save_path_name= '{}/plots/{}{}_target_vs_pred_likelihood_{}'.format(ps_runs_path
                                                                                                                , prefix
                                                                                                                , sigma_str
                                                                                                                , ps_checkpoint_name),
                                                        title='{}{}'.format(prefix, sigma_str),
                                                        input_sequence = inv_norm_or_not(input_sequence),
                                                        **plot_settings
                                                        )
                    plot_distributions(target, pred,
                                       linspace_binning_inv_norm, linspace_binning_max_inv_norm,
                                       title='{}{}'.format(prefix, sigma_str),
                                       save_path_name = '{}/plots/{}{}_distributions_snapshot_{}'.format(ps_runs_path
                                                                                                             ,prefix
                                                                                                             ,sigma_str
                                                                                                             ,ps_checkpoint_name),
                )
            # break
