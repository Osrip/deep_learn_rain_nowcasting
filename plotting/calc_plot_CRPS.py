from helper.calc_CRPS import crps_vectorized, element_wise_crps, iterate_crps_element_wise
from load_data import inverse_normalize_data, invnorm_linspace_binning
import numpy as np
from baselines import LKBaseline
import torch
import torchvision.transforms as T
import einops
import matplotlib.pyplot as plt
from helper.helper_functions import save_zipped_pickle, load_zipped_pickle, img_one_hot
from helper.memory_logging import print_gpu_memory


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
            pred_np = pred.cpu().detach().numpy()
            # We take normalized pred as we already passed the inv normalized binning to the calculate_crps function
            crps_np_model = iterate_crps_element_wise(pred_np, target_inv_normed, linspace_binning_inv_norm,
                                                      linspace_binning_max_inv_norm)


            if not (np.round(crps_model_tc.cpu().numpy(), 4) == np.round(crps_np_model, 4)).all():
                raise ValueError('BUG!!')
            else:
                print('all good')


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

    test = element_wise_crps(bin_probs, observation, bin_edges,)





