from helper.helper_functions import one_hot_to_mm
from load_data import inverse_normalize_data
from baselines import LKBaseline
import torchvision.transforms as T
import torch



def plot_CRPS(model, data_loader, filter_and_normalization_params, linspace_binning_params, plot_settings,
              ps_runs_path, ps_run_name, ps_checkpoint_name, ps_device, ps_inv_normalize,
              ps_gaussian_smoothing_multiple_sigmas, ps_multiple_sigmas, prefix='', **__):

    '''
    In progress...
    '''

    filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized,\
        linspace_binning_max_unnormalized = filter_and_normalization_params

    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params

    if ps_inv_normalize:
        inv_norm_or_not = lambda x: inverse_normalize_data(x, mean_filtered_data, std_filtered_data, inverse_log=True,
                                                           inverse_normalize=True)
    else:
        inv_norm_or_not = lambda x: x

    for i, (input_sequence, target_one_hot, target, _) in enumerate(data_loader):
        input_sequence = input_sequence.to(ps_device)
        model = model.to(ps_device)
        pred = model(input_sequence)

        pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)


def plot_FSS(model, data_loader, filter_and_normalization_params, linspace_binning_params, settings, plot_settings,
             ps_runs_path, ps_run_name, ps_checkpoint_name, ps_device, ps_gaussian_smoothing_multiple_sigmas,
             ps_multiple_sigmas, prefix='', **__):

    '''
    ** expects plot_settings
    Always inv normalizes (independently of ps_inv_normalize) as optical flow cannot operate in inv norm space!
    In progress...
    '''

    filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized,\
        linspace_binning_max_unnormalized = filter_and_normalization_params

    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params


    inv_norm = lambda x: inverse_normalize_data(x, mean_filtered_data, std_filtered_data, inverse_log=True,
                                                           inverse_normalize=True)


    for i, (input_sequence, target_one_hot, target, _) in enumerate(data_loader):
        input_sequence = input_sequence.to(ps_device)
        model = model.to(ps_device)
        pred = model(input_sequence)

        pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)
        pred_mm_inv_normed = inv_norm(pred_mm)

        logging_type = None
        lk_baseline = LKBaseline(logging_type, mean_filtered_data, std_filtered_data, **settings)
        input_sequence_inv_normed = inv_norm(input_sequence).to('cpu')
        pred_mm_baseline, _, _ = lk_baseline(input_sequence_inv_normed)
        pred_mm_baseline = T.CenterCrop(size=32)(pred_mm_baseline)
        pred_mm_baseline = pred_mm_baseline.detach().numpy()










