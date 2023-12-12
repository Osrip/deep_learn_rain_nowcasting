import numpy

from load_data import inverse_normalize_data
import numpy as np
from baselines import LKBaseline
import torch
import torchvision.transforms as T
import einops
import matplotlib.pyplot as plt
from helper.helper_functions import save_zipped_pickle, load_zipped_pickle, img_one_hot
from helper.memory_logging import print_gpu_memory


def crps_vectorized(pred: torch.Tensor, target: torch.Tensor,
                    linspace_binning_inv_norm: np.ndarray, linspace_binning_max_inv_norm: np.ndarray, device, **__):
    '''
    pred: pred_np: binned prediction b x c x h x w
    target: target_inv_normed: target in inv normed space b x h x w
    linspace_binning_inv_norm: left bins edges in inv normed space

    returns CRPS for each pixel in shape b x h x w
    '''
    # Calculations related to binning
    bin_edges_all = np.append(linspace_binning_inv_norm, linspace_binning_max_inv_norm)
    bin_edges_all = torch.from_numpy(bin_edges_all).to(device)
    bin_edges_right = bin_edges_all[1:]
    bin_sizes = torch.diff(bin_edges_all)

    # Unsqueeze binning (adding None dimension) to get same dimensionality as pred with c being dim 1
    bin_edges_right_c_h_w = bin_edges_right[None, :, None, None]
    bin_sizes_unsqueezed_b_c_h_w = bin_sizes[None, :, None, None]

    # in element-wise we are looping through b ins while observation stays constant
    # We are adding a new c dimension to targets where for each target value is replaced by an array of 1s and 0s depending on whether
    # the binning is smaller or bigger than the target
    # heavyside step b x c x h x w --> same as pred
    # target b x h x w
    # binning c

    # Adding c dim that are comparisons to observation: Can also be interpreted as
    # Calculate the heavyside step function (0 below a vertain value 1 above)
    # This repaces if condition in element-wise calculation by just adding
    # heavyside_step is -1 for all bin edges that are on the right side (bigger) than the observation (target)
    target = target[:, None, :, :]
    heavyside_step = (target <= bin_edges_right_c_h_w).float()

    # Calculate CDF
    pred_cdf = torch.cumsum(pred, axis=1)
    # Substract heaviside step
    pred_cdf = pred_cdf - heavyside_step
    # Square
    pred_cdf = torch.square(pred_cdf)
    # Weight according to bin sizes
    pred_cdf = pred_cdf * bin_sizes_unsqueezed_b_c_h_w
    # Sum to get CRPS --> c dim is summed so b x c x h x w --> b x h x w
    crps = torch.sum(pred_cdf, axis=1)

    return crps


def element_wise_crps(bin_probs, observation, bin_edges):
    """
    VIDEO ZUR IMPLEMENTATION IN ICLOUD NOTES UNTER NOTIZ "CRPS"
    Calculate CRPS between an empirical distribution and a point observation.
    Parameters:
    - bin_edges : array-like, bin edges
    !!including leftmost and rightmost edge!!!
    - bin_probs : array-like, probabilities of each bin --> len(bin_probs) == len(bin_edges - 1) as last right bin not included!
    - observation : float, observed value
    Returns:
    - CRPS value : float
    """
    # TODO Speed this up with jit!
    cdf = np.cumsum(bin_probs)
    crps = 0
    # Iterating through each bin and looking whether observation is outside
    for i in range(len(bin_edges)-1):
        left_edge = bin_edges[i]
        right_edge = bin_edges[i+1]
        if observation > right_edge:
            crps += cdf[i] ** 2 * (right_edge - left_edge)
            # Eveything smaller than observation is added to represent integral

        # elif observation < right_edge:
        else:
            crps += (cdf[i] - 1) ** 2 * (right_edge - left_edge)
            # For the bin that the observation is in and all larger bins Observation - 1 is added

    return crps


def calc_CRPS(model, data_loader, filter_and_normalization_params, linspace_binning_params, settings, plot_settings,
             steps_settings,
             ps_runs_path, ps_run_name, ps_checkpoint_name, ps_device, ps_gaussian_smoothing_multiple_sigmas,
             ps_multiple_sigmas, crps_calc_on_every_n_th_batch, crps_load_steps_crps_from_file,
             prefix='', test_output=False, vec_crps=True, **__):

    '''
    This function calculates the mean and std of the FSS over the whole dataset given by the data loader (validations
    set required). The FSS is calculated for different thresholds and scales. The thresholds are given by the
    fss_space_threshold parameter and the scales are given by the fss_linspace_scale parameter. The FSS is calculated
    for each threshold and scale for each sample in the dataset of which the mean and std are then calculated.
    ** expects plot_settings
    Always inv normalizes (independently of ps_inv_normalize) as optical flow cannot operate in inv norm space!
    In progress...

    calc_on_every_n_th_batch=4 <--- Only use every nth batch of data loder to calculate FSS (for speed); at least one
    batch is calculated
    '''
    with torch.no_grad():
        if crps_calc_on_every_n_th_batch > len(data_loader):
            crps_calc_on_every_n_th_batch = len(data_loader)

        filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized,\
            linspace_binning_max_unnormalized = filter_and_normalization_params

        linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params


        inv_norm = lambda x: inverse_normalize_data(x, mean_filtered_data, std_filtered_data, inverse_log=True,
                                                               inverse_normalize=True)


        df_data = []

        preds_and_targets = {}
        preds_and_targets['pred_mm_inv_normed'] = []
        preds_and_targets['pred_mm_steps_baseline'] = []
        preds_and_targets['target_inv_normed'] = []

        logging_type = None
        steps_baseline = LKBaseline(logging_type, mean_filtered_data, std_filtered_data, use_steps=True,
                                    steps_settings=steps_settings, **settings)




        crps_model_list_tc = []
        crps_steps_list_tc = []

        print('CRPS calculations started')
        for i, (input_sequence, target_one_hot, target, _) in enumerate(data_loader):
            print(f'Batch_num: {i}')
            print_gpu_memory()
            if not (i % crps_calc_on_every_n_th_batch == 0):
                break

            input_sequence = input_sequence.to(ps_device)
            target_one_hot = target_one_hot.to(ps_device)
            target = target.to(ps_device)

            model = model.to(ps_device)
            pred = model(input_sequence)

            if ps_gaussian_smoothing_multiple_sigmas:
                pred = pred[0]

            linspace_binning_inv_norm, linspace_binning_max_inv_norm = invnorm_linspace_binning(linspace_binning,
                                                                                                linspace_binning_max,
                                                                                                mean_filtered_data,
                                                                                                std_filtered_data)

            # ! USE INV NORMED PREDICTIONS FROM MODEL ! Baseline is calculated in unnormed space

            if not crps_load_steps_crps_from_file:
                # Baseline Prediction
                input_sequence_inv_normed = inv_norm(input_sequence).to('cpu')
                pred_ensemble_steps_baseline, _, _ = steps_baseline(input_sequence_inv_normed)
                pred_ensemble_steps_baseline = T.CenterCrop(size=settings['s_width_height_target'])(pred_ensemble_steps_baseline)

                # target = target.detach().cpu().numpy()
                target_inv_normed = inv_norm(target)

                # Calculate CRPS for baseline
                pred_ensemble_steps_baseline = pred_ensemble_steps_baseline.cpu().detach().numpy()
                steps_binning_tc = create_binning_from_ensemble(pred_ensemble_steps_baseline, linspace_binning, **settings)
                # steps_binning_np = steps_binning_torch.cpu().detach().numpy()

                # crps_np_steps = iterate_crps_element_wise(steps_binning_np, target_inv_normed, linspace_binning_inv_norm, linspace_binning_max_inv_norm)
                crps_steps_tc = crps_vectorized(steps_binning_tc, target_inv_normed, linspace_binning_inv_norm,
                                             linspace_binning_max_inv_norm, **settings)

                crps_steps_list_tc.append(crps_steps_tc.detach())

            # Calculate CRPS for model predictions

            # if vec_crps:
            # vec_crps
            # target_inv_normed = torch.from_numpy(target_inv_normed)
            crps_model_tc = crps_vectorized(pred, target_inv_normed, linspace_binning_inv_norm,
                                            linspace_binning_max_inv_norm, **settings)

            crps_model_list_tc.append(crps_model_tc.detach())


            # test vectorized with element-wise
            # else:
            # pred_np = pred.cpu().detach().numpy()
            # # We take normalized pred as we already passed the inv normalized binning to the calculate_crps function
            # crps_np_model = iterate_crps_element_wise(pred_np, target_inv_normed, linspace_binning_inv_norm,
            #                                           linspace_binning_max_inv_norm)
            #
            # if not (np.round(crps_vec, 2) == np.round(crps_np_model, 2)).all():
            #     raise ValueError('BUG!!')
            # else:
            #     print('all good')


        save_dir = settings['s_dirs']['logs']
        save_name_model = 'crps_model'
        save_name_steps = 'crps_steps'

        # Saving all arrays
        # They have dimensions sample_num x batch x height x width

        crps_model_all_tc = torch.stack(crps_model_list_tc)
        # crps_model_mean = np.mean(crps_np_model_all)
        # crps_model_std = np.std(crps_np_model_all)
        save_zipped_pickle('{}/{}'.format(save_dir, save_name_model), crps_model_all_tc)

        if not crps_load_steps_crps_from_file:
            crps_steps_all_tc = torch.stack(crps_steps_list_tc)
            # crps_steps_mean = np.mean(crps_np_steps_all)
            # crps_steps_std = np.std(crps_np_steps_all)
            save_zipped_pickle('{}/{}'.format(save_dir, save_name_steps), crps_steps_all_tc)



def create_binning_from_ensemble(ensemble: np.ndarray, linspace_binning, s_num_bins_crossentropy, device, **__):
    '''
    expects **settings kwargs
    Use the NORMALIZED LINSPACE_BINNING

    '''
    #
    num_ensemble_member = ensemble.shape[1]

    ensembles_one_hot = img_one_hot(ensemble, s_num_bins_crossentropy, linspace_binning)
    ensembles_one_hot = ensembles_one_hot.to(device)
    ensembles_one_hot = einops.rearrange(ensembles_one_hot, 'b e w h c -> b e c w h')
    # Dimensions now: batch x ensemble_members x channel (one hot binning) x width x height
    # Summing along ensemble dimension in order to get unnormalized binning
    ensembles_binned_unnormalized = torch.sum(ensembles_one_hot, dim=1) #  tested: looks reasonable
    # Dimensions now: batch x channel (one hot binning) x width x height
    # Normalizing by the number of ensembles
    binning_from_ensemble = ensembles_binned_unnormalized / num_ensemble_member

    # TEST
    # test = torch.sum(ensembles_binned, dim=1)
    # test_np = test.numpy()
    # test_result = test[test == 1].all()

    return binning_from_ensemble


def iterate_crps_element_wise(pred_np, target_inv_normed, linspace_binning_inv_norm, linspace_binning_max_inv_norm):
    '''
    pred_np: binned prediction b x c x h x w
    target_inv_normed: target in inv normed space b x h x w
    linspace_binning_inv_norm: left bins edges in inv normed space
    '''
    calculate_crps_lambda = lambda x, y: element_wise_crps(x, y,
                                                           np.append(linspace_binning_inv_norm, linspace_binning_max_inv_norm),)
    shape_pred = np.shape(pred_np)
    shape_target = np.shape(target_inv_normed)
    if not (shape_target[0] == shape_pred[0] and shape_target[1] == shape_pred[2] and shape_target[2] == shape_pred[3]):
        raise ValueError('Dimensionality mismatch between prediction and target (leaving away channel dimension of prediction')

    crps_out = np.zeros(shape=shape_target)

    for b in range(shape_target[0]):
        for h in range(shape_target[1]):
            for w in range(shape_target[2]):
                crps = calculate_crps_lambda(pred_np[b, :, h, w], target_inv_normed[b, h, w])
                crps_out[b, h, w] = crps
    return crps_out


def invnorm_linspace_binning(linspace_binning, linspace_binning_max, mean_filtered_data, std_filtered_data):
    '''
    Inverse normalizes linspace binning
    By default the linspace binning only includes the lower bounds#
    Therefore the highest upper bound is missing which is given by linspace_binning_max
    '''
    linspace_binning_inv_norm = inverse_normalize_data(np.array(linspace_binning), mean_filtered_data, std_filtered_data)
    linspace_binning_max_inv_norm = inverse_normalize_data(np.array(linspace_binning_max), mean_filtered_data, std_filtered_data, inverse_log=True)
    return linspace_binning_inv_norm, linspace_binning_max_inv_norm.item()


# def plot_crps(s_dirs, crps_load_steps_crps_from_file, crps_steps_file_path, **__):
#
#     save_dir = s_dirs['logs']
#     save_name_model = 'crps_model'
#     save_name_steps = 'crps_steps'
#
#     crps_model_path = '{}/{}'.format(save_dir, save_name_model)
#
#     if crps_load_steps_crps_from_file:
#         crps_steps_path = crps_steps_file_path
#     else:
#         crps_steps_path = '{}/{}'.format(save_dir, save_name_steps)
#
#     model_name = 'Model'
#     steps_name = 'STEPS'
#
#     model_array = load_zipped_pickle(crps_model_path)
#     steps_array = load_zipped_pickle(crps_steps_path)
#
#     data = [model_array, steps_array]
#     means = [np.mean(array) for array in data]
#     stds = [np.std(array) for array in data]
#     medians = [np.median(array) for array in data]
#
#     # Creating the violin plot with mean as a point and std as a line
#     plt.figure(figsize=(10, 6))
#     ax = sns.violinplot(data=data)
#     ax.set_xticklabels([model_name, steps_name])  # Setting custom names for the x-axis
#
#     # # Plotting mean as a point
#     # ax.scatter(range(len(data)), means, color='red', marker='o', label='Mean')
#     #
#     # # Plotting std as a line
#     # for i in range(len(data)):
#     #     ax.plot([i, i], [means[i] - stds[i], means[i] + stds[i]], color='blue', lw=2, label='STD' if i == 0 else "")
#
#     # Adding text for mean, std, and median values above the plot
#     for i in range(len(data)):
#         plt.text(i, ax.get_ylim()[1] * 1.1, f'Mean: {means[i]:.3f}\nSTD: {stds[i]:.3f}\nMedian: {medians[i]:.3f}',
#                  horizontalalignment='center', size='small', color='black', weight='semibold')
#
#     plt.title('Violin Plots with Mean and STD')
#     plt.legend()
#
#     plt.savefig(f"{s_dirs['plot_dir']}/crps.png", bbox_inches='tight')
#     plt.show()
#     plt.close()  # Close the plot to free memory


def plot_crps(s_dirs, crps_load_steps_crps_from_file, crps_steps_file_path, **__):
    save_dir = s_dirs['logs']
    save_name_model = 'crps_model'
    save_name_steps = 'crps_steps'

    crps_model_path = f'{save_dir}/{save_name_model}'

    if crps_load_steps_crps_from_file:
        crps_steps_path = crps_steps_file_path
    else:
        crps_steps_path = f'{save_dir}/{save_name_steps}'

    model_name = 'Model'
    steps_name = 'STEPS'

    model_crps_tc = load_zipped_pickle(crps_model_path)
    steps_crps_tc = load_zipped_pickle(crps_steps_path)

    model_crps_array = model_crps_tc.detach().cpu().numpy()
    steps_crps_array = steps_crps_tc.detach().cpu().numpy()


    model_crps_array = model_crps_array.flatten()
    steps_crps_array = steps_crps_array.flatten()

    data = [model_crps_array, steps_crps_array]
    means = [np.mean(array) for array in data]
    stds = [np.std(array) for array in data]
    medians = [np.median(array) for array in data]

    plt.figure(figsize=(10, 6))
    parts = plt.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # Adding mean as a point
    plt.scatter(range(1, len(data) + 1), means, color='green', marker='o', label='Mean')

    # Adding std as a line
    for i in range(len(data)):
        plt.plot([i + 1, i + 1], [means[i] - stds[i], means[i] + stds[i]], color='blue', lw=2, label='STD' if i == 0 else "")

    # Adding custom names for the x-axis
    plt.xticks([1, 2], [model_name, steps_name])

    # Adding text for mean, std, and median values above the plot
    for i in range(len(data)):
        plt.text(i + 1, max(data[i]) * 1.1, f'Mean: {means[i]:.3f}\nSTD: {stds[i]:.3f}\nMedian: {medians[i]:.3f}',
                 horizontalalignment='center', size='small', color='black', weight='semibold')

    plt.title('CRPS')
    plt.legend()

    plt.savefig(f"{s_dirs['plot_dir']}/crps.png", bbox_inches='tight')
    plt.show()
    plt.close()  # Close the plot to free memory


if __name__ == '__main__':
    bin_edges = np.array([0, 1, 2, 3, 4])
    bin_probs = np.array([0, 0, 1, 0])
    observation = 2.5

    test = element_wise_crps(bin_probs, observation, bin_edges, )
    pass





