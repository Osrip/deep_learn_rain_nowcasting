import numpy

from load_data import inverse_normalize_data
import numpy as np
from baselines import LKBaseline
import torch
import torchvision.transforms as T
import einops
import matplotlib.pyplot as plt
from helper.helper_functions import save_zipped_pickle, load_zipped_pickle, img_one_hot


def crps_vectorized(pred: torch.Tensor, target: torch.Tensor,
                    linspace_binning_inv_norm: np.ndarray, linspace_binning_max_inv_norm: np.ndarray):
    '''
    pred: pred_np: binned prediction b x c x h x w
    target: target_inv_normed: target in inv normed space b x h x w
    linspace_binning_inv_norm: left bins edges in inv normed space

    returns CRPS for each pixel in shape b x h x w
    '''
    # Calculations related to binning
    bins_edges_all = np.append(linspace_binning_inv_norm, linspace_binning_max_inv_norm)
    bin_edges_all = torch.from_numpy(bins_edges_all)
    bin_edges_right = bin_edges_all[1:]
    bin_sizes = torch.diff(bin_edges_all)

    # Unsqueeze binning (adding None dimension) to get same dimensionality as pred with c being dim 1
    bin_edges_right_c_h_w = bin_edges_right[None, :, None, None]
    bin_sizes_unsqueezed_b_c_h_w = bin_sizes[None, :, None, None]

    # Calculate the heavyside step function (0 below a vertain value 1 above)
    # This repaces if condition in element-wise calculation by just adding
    # heavyside_step is -1 for all bin edges that are on the right side (bigger) than the observation (target)

    '''
    Dim mismatch!
    in element-wise we are looping through b ins while observation stays constant
    We are adding a new c dimension to targets where for each target value is replaced by an array of 1s and 0s depending on whether 
    the binning is smaller or bigger than the target
    heavyside step b x c x h x w --> same as pred
    target b x h x w 
    binning c
    '''
    # TODO: Does this solve it correctly?
    target = target[:, None, :, :]
    heavyside_step = (target <= bin_edges_right_c_h_w).float()

    # Calculate CDF
    pred_cdf = torch.cumsum(pred, axis=1)
    # Substract heaviside step
    pred_cdf = pred_cdf - heavyside_step
    # Square
    pred_cdf = torch.square(pred_cdf, axis=1)
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

    if crps_calc_on_every_n_th_batch < len(data_loader):
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




    crps_np_model_list = []
    crps_np_steps_list = []

    for i, (input_sequence, target_one_hot, target, _) in enumerate(data_loader):
        if not (i % crps_calc_on_every_n_th_batch == 0):
            break

        input_sequence = input_sequence.to(ps_device)
        model = model.to(ps_device)
        pred = model(input_sequence)

        if ps_gaussian_smoothing_multiple_sigmas:
            pred = pred[0].detach().cpu()

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

            target = target.detach().cpu().numpy()
            target_inv_normed = inv_norm(target)

            # Calculate CRPS for baseline
            pred_ensemble_steps_baseline = pred_ensemble_steps_baseline.cpu().detach().numpy()
            steps_binning_torch = create_binning_from_ensemble(pred_ensemble_steps_baseline, linspace_binning, **settings)
            steps_binning_np = steps_binning_torch.cpu().detach().numpy()

            crps_np_steps = iterate_crps(steps_binning_np, target_inv_normed, linspace_binning_inv_norm, linspace_binning_max_inv_norm)
            crps_np_steps_list.append(crps_np_steps)

        # Calculate CRPS for model predictions

        if vec_crps:
            target_inv_normed = torch.from_numpy(target_inv_normed)
            crps_np_model = crps_vectorized(pred, target_inv_normed, linspace_binning_inv_norm,
                                            linspace_binning_max_inv_norm)
        else:
            pred_np = pred.cpu().detach().numpy()
            # We take normalized pred as we already passed the inv normalized binning to the calculate_crps function
            crps_np_model = iterate_crps(pred_np, target_inv_normed, linspace_binning_inv_norm,
                                         linspace_binning_max_inv_norm)
        crps_np_model_list.append(crps_np_model)

    save_dir = settings['s_dirs']['logs']
    save_name_model = 'crps_model'
    save_name_steps = 'crps_steps'

    # Saving all arrays
    # They have dimensions sample_num x batch x height x width

    crps_np_model_all = np.array(crps_np_model_list)
    # crps_model_mean = np.mean(crps_np_model_all)
    # crps_model_std = np.std(crps_np_model_all)
    save_zipped_pickle('{}/{}'.format(save_dir, save_name_model), crps_np_model_all)

    if not crps_load_steps_crps_from_file:
        crps_np_steps_all = np.array(crps_np_steps_list)
        # crps_steps_mean = np.mean(crps_np_steps_all)
        # crps_steps_std = np.std(crps_np_steps_all)
        save_zipped_pickle('{}/{}'.format(save_dir, save_name_steps), crps_np_steps_all)

    if test_output:
        return np.mean(crps_np_model_all), np.std(crps_np_model_all), np.mean(crps_np_steps_all), np.std(crps_np_steps_all)


def create_binning_from_ensemble(ensemble: np.ndarray, linspace_binning, s_num_bins_crossentropy, **__):
    '''
    expects **settings kwargs
    Use the NORMALIZED LINSPACE_BINNING

    '''
    #
    num_ensemble_member = ensemble.shape[1]

    ensembles_one_hot = img_one_hot(ensemble, s_num_bins_crossentropy, linspace_binning)
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


def iterate_crps(pred_np, target_inv_normed, linspace_binning_inv_norm, linspace_binning_max_inv_norm):
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

    model_array = load_zipped_pickle(crps_model_path)
    steps_array = load_zipped_pickle(crps_steps_path)

    model_array = model_array.flatten()
    steps_array = steps_array.flatten()

    data = [model_array, steps_array]
    means = [np.mean(array) for array in data]
    stds = [np.std(array) for array in data]
    medians = [np.median(array) for array in data]

    plt.figure(figsize=(10, 6))
    parts = plt.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    # Adding mean as a point
    plt.scatter(range(1, len(data) + 1), means, color='red', marker='o', label='Mean')

    # Adding std as a line
    for i in range(len(data)):
        plt.plot([i + 1, i + 1], [means[i] - stds[i], means[i] + stds[i]], color='blue', lw=2, label='STD' if i == 0 else "")

    # Adding custom names for the x-axis
    plt.xticks([1, 2], [model_name, steps_name])

    # Adding text for mean, std, and median values above the plot
    for i in range(len(data)):
        plt.text(i + 1, max(data[i]) * 1.1, f'Mean: {means[i]:.3f}\nSTD: {stds[i]:.3f}\nMedian: {medians[i]:.3f}',
                 horizontalalignment='center', size='small', color='black', weight='semibold')

    plt.title('Violin Plots with Mean and STD')
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
















'''
Write test:

I want to test my CRPS function. Can you write a function that creates test data? I would to create a random_test_data_set. target_inv_normed contain the direct values of the initial random data set.  target_inv_normed contains one sample and I want the samples to be an adjustable variable
target np should have dimensions b x h x w so batch (adjustable varibale as well) x 32 x 32

For pred_np samples contained in target_inv_normed should have an added gaussian blurring. The magnitude of the standard deviation as well as the magitude should be varied for each std_modified_data_set (containing a several std modified test_data_set s). Additionally there should be an offset_modified_data_set (containining several offset modified std_modified_data_set s)  created with an added offset (just adding a number to each value to create offset). This way the test data has two variations from the inititial random test data: std gaussian blurring adding and offset adding.  

Now pred_np has a probability binning for each pixel. It has dimensions b x c x h x w where c is the binning dimension, thus b x 64 x 32 x 32 thus 64 bins. Use the std that was used to modify the current set within the std_modified_data_set to create the binning around the mean of the current pixel.

Here are the functions to be tested:

def iterate_crps(pred_np, target_inv_normed, linspace_binning_inv_norm, linspace_binning_max_inv_norm):

    # pred_np: binned prediction b x c x h x w
    # target_inv_normed: target in inv normed space b x h x w
    # linspace_binning_inv_norm: left bins edges in inv normed space

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

linspace_binning_inv_norm =
[-0.02084237  0.01762839  0.05761065  0.0991638   0.14234956  0.18723208,  0.23387801  0.28235665  0.33273999  0.38510288  0.43952308  0.49608144,  0.55486196  0.61595194  0.67944213  0.74542682  0.81400403  0.88527561,  0.95934743  1.03632951  1.11633618  1.19948629  1.28590334  1.37571568,  1.46905672  1.56606509  1.66688489  1.77166587  1.88056365  1.99373999,  2.11136298  2.23360735  2.36065465  2.49269359  2.6299203   2.7725386,  2.92076032  3.07480563  3.23490331  3.40129118  3.57421637  3.75393573,  3.94071621  4.13483522  4.3365811   4.5462535   4.76416386  4.99063584,  5.22600583  5.47062342  5.72485195  5.98906903  6.26366711  6.54905405,  6.84565374  7.15390673  7.47427087  7.80722201  8.15325468  8.51288286,  8.88664069  9.27508335  9.67878778 10.09835361]

linspace_binning_max_inv_norm =
10.534404044730364

I would like to end up with a line plot where offset is on the x axis and the CRPS is on the y axis (mean represents line and std is a shaded area around the line in the same color as the line)
Different stds correspond to different colors of the lines.
Let's start with 30 different offsets and 5 different means.

'''