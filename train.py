import torch
import torch.nn as nn
import torchvision.transforms as T
from modules_blocks import Network
import datetime
from load_data import PrecipitationFilteredDataset, inverse_normalize_data, filtering_data_scraper, lognormalize_data, random_splitting_filtered_indecies
from torch.utils.data import DataLoader

import numpy as np
from helper.helper_functions import load_zipped_pickle, save_zipped_pickle, one_hot_to_mm, save_whole_project
import os
from plotting_list_based.plot_img_histogram import plot_img_histogram
from plotting.plot_images import plot_target_vs_pred, plot_target_vs_pred_with_likelihood
from plotting_list_based.plot_quality_metrics import plot_mse_light, plot_mse_heavy, plot_losses, plot_average_preds, plot_pixelwise_preds
import warnings
from tests.test_basic_functions import test_all
from hurry.filesize import size
from tqdm import tqdm
import psutil

# from pysteps import verification
# fss = verification.get_method("FSS")

mse_loss = torch.nn.MSELoss()


def print_gpu_memory():
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("GPU Memory: Total: {}, Free: {}, Used:{}".format(size(info.total), size(info.free), size(info.used)))


def print_ram_usage():
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used:', size(psutil.virtual_memory()[3]))



def validate(model, validation_data_loader, linspace_binning, linspace_binning_max, linspace_binning_min,
             epoch, mean_filtered_data, std_filtered_data, s_width_height_target, device, s_plot_target_vs_pred_boo,**__):

    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        validation_losses = []
        # val_mse_persistence_target = []
        val_mse_zeros_target = []
        val_mse_model_target = []


        for i, (input_sequence, target_one_hot, target) in enumerate(validation_data_loader):
            input_sequence = input_sequence.float()
            input_sequence = input_sequence.to(device)
            # Cropping is already done in data loader
            # target_one_hot = T.CenterCrop(size=s_width_height_target)(target_one_hot)
            target_one_hot = target_one_hot.float()
            target_one_hot = target_one_hot.to(device)

            pred = model(input_sequence)
            loss = criterion(pred, target_one_hot)
            loss = float(loss.detach().cpu().numpy())


            validation_losses.append(loss)

            # Calculatatie verfification metrics

            target = target.detach().to(device)
            pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)
            pred_mm = torch.from_numpy(pred_mm).to(device).detach()

            if i == 0:
                # Imshow plots
                if s_plot_target_vs_pred_boo:
                    inv_norm = lambda x: inverse_normalize_data(x, mean_filtered_data, std_filtered_data, inverse_log=False,
                                                                inverse_normalize=True)

                    plot_target_vs_pred(inv_norm(target), inv_norm(pred_mm), vmin=inv_norm(linspace_binning_min),
                                        vmax=inv_norm(linspace_binning_max),
                                        save_path_name='{}/ep{:04}_VAL_target_vs_pred'.format(s_dirs['plot_dir_images'], epoch),
                                        title='Validation data (log, not normalized)')

                    plot_target_vs_pred_with_likelihood(inv_norm(target), inv_norm(pred_mm), pred,
                                        linspace_binning=inv_norm(linspace_binning), vmin=inv_norm(linspace_binning_min),
                                        vmax=inv_norm(linspace_binning_max),
                                        save_path_name='{}/ep{:04}_VAL_target_vs_pred_likelihood'.format(s_dirs['plot_dir_images'], epoch),
                                        title='Validation data (log, not normalized)')

            # val_mse_persistence_target.append(mse_loss(cropped_persistence, target).item())
            val_mse_zeros_target.append(mse_loss(torch.zeros(target.shape).to(device), target).item())
            val_mse_model_target.append(mse_loss(pred_mm, target).item())


    return validation_losses, val_mse_zeros_target, val_mse_model_target


def train(model, s_sim_name, device, s_learning_rate: int, s_num_epochs: int, s_num_input_time_steps: int, s_num_lead_time_steps,
          s_num_bins_crossentropy, s_width_height_target, s_batch_size, s_ratio_training_data,
          s_dirs, s_local_machine_mode, s_log_transform, s_normalize, s_plot_average_preds_boo, s_plot_pixelwise_preds_boo,
          s_save_trained_model, s_data_loader_chunk_size, s_plot_target_vs_pred_boo, s_plot_img_histogram_boo, s_plot_losses_boo,
          s_plot_mse_boo, settings, **__):

    if s_log_transform:
        transform_f = lambda x: np.log(x + 1)
    else:
        transform_f = lambda x: x

    # save_settings(settings, s_dirs['save_dir'])

    # try:
    save_whole_project(s_dirs['code_dir'])
    # except Exception:
    #     print('Could not create backup copy of code')
    # num_pictures_loaded = s_num_input_time_steps + 1 + s_num_lead_time_steps

    # relative index of last input picture (starting from first input picture as idx 1)
    last_input_rel_idx = s_num_input_time_steps
    #  relative index of target picture (starting from first input picture as idx 1)
    target_rel_idx = s_num_input_time_steps + 1 + s_num_lead_time_steps
    # Number of pictures

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=s_learning_rate)
    # TODO: Enable saving to pickle at some point
    print('Load Data', flush=True)

    filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized, linspace_binning_max_unnormalized =\
        filtering_data_scraper(transform_f=transform_f, last_input_rel_idx=last_input_rel_idx, target_rel_idx=target_rel_idx,
                               **settings)

    # Normalize linspace binning thresholds now that data is available
    linspace_binning_min = lognormalize_data(linspace_binning_min_unnormalized, mean_filtered_data, std_filtered_data,
                                             transform_f, s_normalize) - 0.00001
    # Subtract a small number to account for rounding errors made in the normalization process
    linspace_binning_max = lognormalize_data(linspace_binning_max_unnormalized, mean_filtered_data, std_filtered_data,
                                             transform_f, s_normalize)

    linspace_binning_min = linspace_binning_min - 0.1
    linspace_binning_max = linspace_binning_max + 0.1

    linspace_binning = np.linspace(linspace_binning_min, linspace_binning_max, num=s_num_bins_crossentropy,
                                   endpoint=False)  # num_indecies + 1 as the very last entry will never be used

    # Defining and splitting into training and validation data set
    num_training_samples = int(len(filtered_indecies) * s_ratio_training_data)
    num_validation_samples = len(filtered_indecies) - num_training_samples

    filtered_indecies_training, filtered_indecies_validation = random_splitting_filtered_indecies(
        filtered_indecies, num_training_samples, num_validation_samples, s_data_loader_chunk_size)

    train_data_set = PrecipitationFilteredDataset(filtered_indecies_training, mean_filtered_data, std_filtered_data,
                                                  linspace_binning_min, linspace_binning_max, linspace_binning, transform_f, **settings)

    validation_data_set = PrecipitationFilteredDataset(filtered_indecies_validation, mean_filtered_data, std_filtered_data,
                                                  linspace_binning_min, linspace_binning_max, linspace_binning, transform_f, **settings)


    # full_data_set = PrecipitationFilteredDataset(filtered_indecies, mean_filtered_data, std_filtered_data,
    #                                               linspace_binning_min, linspace_binning_max, linspace_binning, transform_f, **settings)
    #

    # train_data_set, validation_data_set = torch.utils.data.random_split(full_data_set,
    #                                                                     [num_training_samples, num_validation_samples])

    train_data_loader = DataLoader(train_data_set, batch_size=s_batch_size, shuffle=True, drop_last=True)
    del train_data_set

    validation_data_loader = DataLoader(validation_data_set, batch_size=s_batch_size, shuffle=True, drop_last=True)
    del validation_data_set
    print('Size data set: {} \nof which training samples: {}  \nvalidation samples: {}'.format(len(filtered_indecies),
                                                                                                 num_training_samples,
                                                                                                 num_validation_samples))
    print('Num training batches: {} \nNum validation Batches: {} \nBatch size: {}'.format(len(train_data_loader),
                                                                                       len(validation_data_loader),
                                                                                       s_batch_size))
    losses = []

    validation_losses = []
    val_mses_persistence_target = []
    val_mses_zeros_target = []
    val_mses_model_target = []

    relative_mses = []
    persistence_target_mses = []
    zeros_target_mses = []
    model_target_mses = []

    if s_plot_average_preds_boo or s_plot_pixelwise_preds_boo:
        all_pred_mm = []
        all_target_mm = []

    # Iterate through epochs
    for epoch in range(s_num_epochs):
        if device.type == 'cuda':
            print_gpu_memory()
        print_ram_usage()

        inner_losses = []
        inner_relative_mses = []
        inner_persistence_target_mses = []
        inner_zeros_target_mses = []
        inner_model_target_mses = []

        # Iterate through batches
        for i, (input_sequence, target_one_hot, target) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):

            # Linspace binning is processed by the data loader and is therefore converted to e tensor with added batch
            # dimensions
            # convert linspace_binning back to 1D array
            # linspace_binning = np.array(linspace_binning[0])
            linspace_binning_control = linspace_binning
            if i > 0:
                if not (linspace_binning_control == linspace_binning).all():
                    raise Exception('There is something wrong with linspace binning!')
                    warnings.warn('There is something wrong with linspace binning!')
                    # print('There is something wrong with linspace binning!')
            input_sequence = input_sequence.float()

            input_sequence = input_sequence.to(device)
            input_sequence = input_sequence.to(device=device, dtype=torch.float32)
            target_one_hot = target_one_hot.to(device)

            # todo --> into getitem
            # Cropping is already done in data loader!
            # target = T.CenterCrop(size=s_width_height_target)(target)
            # target_one_hot = T.CenterCrop(size=s_width_height_target)(target_one_hot)
            target_one_hot = target_one_hot.float()

            target_one_hot = target_one_hot.to(device)
            target = target.to(device).detach()

            ####### Forward, Backward pass ##########
            optimizer.zero_grad()
            pred = model(input_sequence)
            loss = criterion(pred, target_one_hot)
            loss.backward()
            optimizer.step()
            ################################

            # Quality metrics
            pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)
            pred_mm = torch.from_numpy(pred_mm).to(device).detach()

            if s_plot_average_preds_boo or s_plot_pixelwise_preds_boo:
                for k in range(pred_mm.shape[0]):

                    pred_mm_arr = pred_mm.detach().cpu().numpy()
                    pred_mm_arr = inverse_normalize_data(pred_mm_arr, mean_filtered_data, std_filtered_data)
                    all_pred_mm.append(pred_mm_arr[k, :, :])

                    target_arr = target.detach().cpu().numpy()
                    target_arr = inverse_normalize_data(target_arr, mean_filtered_data, std_filtered_data)
                    all_target_mm.append(target_arr[k, :, :])

            inner_loss = float(loss.detach().cpu().numpy())
            # TODO: convert cudo tensor to numpy!! How to push to CPU for numpy conversion and then back to cuda??
            # loss_copy = copy.deepcopy(loss)
            # inner_loss = float(loss_copy.detach().to('cpu').numpy())
            # del loss_copy
            # loss = loss.to(device)
            inner_losses.append(inner_loss)

            persistence = input_sequence[:, -1, :, :]
            persistence = T.CenterCrop(size=s_width_height_target)(persistence)
            mse_persistence_target = mse_loss(persistence, target).item()
            # TODO REMOVE THIS ONLY DEBUGGING:
            mse_zeros_target = mse_loss(torch.zeros(target.shape).to(device), target).item()
            # mse_persistence_target = mse_loss(mse_persistence_target, target).item()

            mse_model_target = mse_loss(pred_mm, target).item()
            relative_mse = 1 - mse_persistence_target / mse_model_target
            inner_relative_mses.append(relative_mse)
            inner_persistence_target_mses.append(mse_persistence_target)
            inner_zeros_target_mses.append(mse_zeros_target)
            inner_model_target_mses.append(mse_model_target)

            # TODO: Currently Target is baseline for test purposes. Change that for obvious reasons!!!
            if i == 0:

                inv_norm = lambda x: inverse_normalize_data(x, mean_filtered_data, std_filtered_data, inverse_log=False,
                                                            inverse_normalize=True)

                if s_plot_target_vs_pred_boo:
                    plot_target_vs_pred(inv_norm(target), inv_norm(pred_mm), vmin=inv_norm(linspace_binning_min),
                                        vmax=inv_norm(linspace_binning_max),
                                        save_path_name='{}/ep{:04}_TRAIN_target_vs_pred'.format(s_dirs['plot_dir_images'], epoch),
                                        title='Training data (log, not normalized)')

                    plot_target_vs_pred_with_likelihood(inv_norm(target), inv_norm(pred_mm), pred,
                                                        linspace_binning=inv_norm(linspace_binning),
                                                        vmin=inv_norm(linspace_binning_min),
                                                        vmax=inv_norm(linspace_binning_max),
                                                        save_path_name='{}/ep{:04}_TRAIN_target_vs_pred_likelihood'.format(
                                                            s_dirs['plot_dir_images'], epoch),
                                                        input_sequence=input_sequence,
                                                        title='Validation data (log, not normalized)')

        relative_mses.append(inner_relative_mses)
        persistence_target_mses.append(inner_persistence_target_mses)
        zeros_target_mses.append(inner_zeros_target_mses)
        model_target_mses.append(inner_model_target_mses)
        losses.append(inner_losses)

        avg_inner_loss = np.mean(inner_losses)

        inner_validation_losses, inner_val_mse_zeros_target, inner_val_mse_model_target\
            = validate(model, validation_data_loader, linspace_binning, linspace_binning_max, linspace_binning_min,
             epoch, mean_filtered_data, std_filtered_data, **settings)

        val_mses_zeros_target.append(inner_val_mse_zeros_target)
        val_mses_model_target.append(inner_val_mse_model_target)

        validation_losses.append(inner_validation_losses)

        avg_inner_validation_loss = np.mean(inner_validation_losses)

        print('Epoch: {} Training loss: {}, Validation loss: {}'.format(epoch, avg_inner_loss, avg_inner_validation_loss),
              flush=True)
        if s_plot_mse_boo:
            plot_mse_light([relative_mses], label_list=['relative MSE'],
                     save_path_name='{}/relative_mse'.format(s_dirs['plot_dir'], epoch),
                     title='Relative MSE on lognorm data')

            # plot_mse([persistence_target_mses, zeros_target_mses, model_target_mses],
            #          label_list=['Persistence Target MSE', 'Zeros target MSEs', 'Model Target MSE'],
            #          save_path_name='{}/mse'.format(s_dirs['plot_dir'], epoch),
            #          title='MSE on lognorm data')

            # TODO: Why does zeros_target_mses differ that much from val_mses_model_target ???
            plot_mse_heavy(mses_list=[persistence_target_mses, zeros_target_mses, model_target_mses,
                      val_mses_zeros_target, val_mses_model_target],
                     label_list=['Persistence Target MSE', 'Zeros target MSEs', 'Model Target MSE',
                                 'VAL Zeros target MSEs', 'VAL Model Target MSE'],
                           color_list=['g', 'y', 'b', 'y', 'b'], linestyle_list=['-', '-', '-', '--', '--'],
                     save_path_name='{}/mse_with_val'.format(s_dirs['plot_dir'], epoch),
                     title='MSE on lognorm data')

            plot_mse_heavy(mses_list=[persistence_target_mses, zeros_target_mses, model_target_mses,
                     val_mses_model_target],
                     label_list=['Persistence Target MSE', 'Zeros target MSEs', 'Model Target MSE',
                                 'VAL Model Target MSE'],
                           color_list=['g', 'y', 'b', 'b'], linestyle_list=['-', '-', '-', '--'],
                     save_path_name='{}/mse_with_val_training_baseline'.format(s_dirs['plot_dir'], epoch),
                     title='MSE on lognorm data')
        if s_plot_losses_boo:
            plot_losses(losses, validation_losses, save_path_name='{}/{}_loss.png'.format(s_dirs['plot_dir'], s_sim_name))
        if s_plot_img_histogram_boo:
            plot_img_histogram(pred_mm, '{}/ep{:04}_pred_dist'.format(s_dirs['plot_dir'], epoch), linspace_binning_min,
                               linspace_binning_max, ignore_min_max=False, title='Prediciton', **settings)
            plot_img_histogram(input_sequence, '{}/ep{:04}_input_dist'.format(s_dirs['plot_dir'], epoch),
                               linspace_binning_min, linspace_binning_max, title='Input', **settings)
            plot_img_histogram(target, '{}/ep{:04}_target_dist'.format(s_dirs['plot_dir'], epoch),
                               linspace_binning_min, linspace_binning_max, ignore_min_max=False, title='Target', **settings)

        if s_plot_average_preds_boo:
            plot_average_preds(all_pred_mm, all_target_mm, len(train_data_loader)*s_batch_size, '{}/average_preds'.
                               format(s_dirs['plot_dir']))

        if s_plot_pixelwise_preds_boo:
            try:
                plot_pixelwise_preds(all_pred_mm, all_target_mm, epoch, '{}/pixelwise_preds'.format(s_dirs['plot_dir']))
            except ValueError:
                warnings.warn('Could not plot pixel wise pres plot in epoch {}'.format(epoch))

        if s_save_trained_model:
            save_zipped_pickle('{}/model_epoch_{}'.format(s_dirs['model_dir'], epoch), model)

    return model


def main(s_save_trained_model, s_load_model, s_num_input_time_steps, s_upscale_c_to, s_load_model_name, s_width_height,
         s_num_bins_crossentropy, s_testing, settings, **_):
    if s_testing:
        test_all()

    if s_load_model:
        model = load_zipped_pickle('runs/{}/model/trained_model'.format(s_load_model_name))
    else:
        model = Network(c_in=s_num_input_time_steps, s_upscale_c_to=s_upscale_c_to,
                        s_num_bins_crossentropy=s_num_bins_crossentropy, s_width_height_in=s_width_height)
    model = model.to(device)
    # NETWORK STILL NEEDS NUMBER OF OUTPUT CHANNELS num_channels_one_hot_output !!!

    # Throws an error on remote venv for some reason
    # optimized_model = check_backward(model, s_learning_rate=0.001, device='cpu')


    print('Model has {} parameters'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('Training started on {}'.format(device), flush=True)

    trained_model = train(model=model, settings=settings, **settings)


if __name__ == '__main__':
    #  Training data
    # num_training_samples = 20  # 1000  # Number of loaded pictures (first pics not used for training but only input)
    # num_validation_samples = 20  # 600

    # train_start_date_time = datetime.datetime(2020, 12, 1)
    # s_folder_path = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/weather_data/dwd_datensatz_bits/rv_recalc/RV_RECALC/hdf/'

    s_local_machine_mode = False

    s_sim_name_suffix = '_6_months_train_no_lightning'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        import nvidia_smi
        nvidia_smi.nvmlInit()
    # device = 'cpu'

    if s_local_machine_mode:
        s_sim_name = 'Run_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        s_sim_name = 'Run_{}_ID_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), int(os.environ['SLURM_JOB_ID']))  # SLURM_ARRAY_TASK_ID

    s_dirs = {}
    s_dirs['save_dir'] = 'runs/{}{}'.format(s_sim_name, s_sim_name_suffix)
    s_dirs['plot_dir'] = '{}/plots'.format(s_dirs['save_dir'])
    s_dirs['plot_dir_images'] = '{}/images'.format(s_dirs['plot_dir'])
    s_dirs['model_dir'] = '{}/model'.format(s_dirs['save_dir'])
    s_dirs['code_dir'] = '{}/code'.format(s_dirs['save_dir'])

    for _, make_dir in s_dirs.items():
        if not os.path.exists(make_dir):
            os.makedirs(make_dir)

    settings =\
        {
            's_local_machine_mode': s_local_machine_mode,
            's_sim_name': s_sim_name,
            's_sim_same_suffix': s_sim_name_suffix,

            's_folder_path': '/mnt/qb/butz/bst981/weather_data/dwd_nc/rv_recalc_months/rv_recalc_months',
            's_data_file_names': ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(6)],  # ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(9)],# ['RV_recalc_data_2019-01.nc'], # ['RV_recalc_data_2019-01.nc', 'RV_recalc_data_2019-02.nc', 'RV_recalc_data_2019-03.nc'], #   # ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(9)],
            's_data_variable_name': 'RV_recalc',
            's_choose_time_span': False,
            's_time_span': (datetime.datetime(2020, 12, 1), datetime.datetime(2020, 12, 1)),
            's_ratio_training_data': 0.6,
            's_data_loader_chunk_size': 20,

            # Parameters that give the network architecture
            's_upscale_c_to': 32, #64, #128, # 512,
            's_num_bins_crossentropy': 64,

            # 'minutes_per_iteration': 5,
            's_width_height': 256,
            's_width_height_target': 32,
            's_learning_rate': 0.0001,  # Schedule this at some point??
            's_num_epochs': 1000,
            's_num_input_time_steps': 4,  # The number of subsequent time steps that are used for one predicition
            's_num_lead_time_steps': 1,  # 5, # The number of pictures that are skipped from last input time step to target, starts with 0
            's_optical_flow_input': False,  # Not yet working!
            's_batch_size': 55,  # batch size 22: Total: 32G, Free: 6G, Used:25G | Batch size 26: Total: 32G, Free: 1G, Used:30G --> vielfache von 8 am besten
            's_save_trained_model': True, # saves model every epoch
            's_load_model': False,
            's_load_model_name': 'Run_·20230220-191041',
            's_dirs': s_dirs,
            'device': device,

            # Log transform input/ validation data --> log binning --> log(x+1)
            's_log_transform': True,
            's_normalize': True,

            's_min_rain_ratio_target': 0.01, #Deactivated  # The minimal amount of rain required in the 32 x 32 target for target and its
            # prior input sequence to make it through the filter into the training data

            's_testing': True,

            # Plotting stuff
            's_no_plotting': False, # This sets all plotting boos below to False
            's_plot_average_preds_boo': True,
            's_plot_pixelwise_preds_boo': True,
            's_plot_target_vs_pred_boo': True,
            's_plot_mse_boo': True,
            's_plot_losses_boo': True,
            's_plot_img_histogram_boo': True,

        }

    if settings['s_no_plotting']:
        for en in ['s_plot_average_preds_boo', 's_plot_pixelwise_preds_boo', 's_plot_target_vs_pred_boo', 's_plot_mse_boo',
                   's_plot_losses_boo', 's_plot_img_histogram_boo']:
            settings[en] = False

    if settings['s_local_machine_mode']:
        # settings['s_data_variable_name'] = 'WN_forecast'
        settings['s_data_variable_name'] = 'RV_recalc'

        # settings['s_folder_path'] = 'dwd_nc/test_data'
        # settings['s_folder_path'] = '/mnt/common/Jan/Programming/weather_data/dwd_nc/rv_recalc_months'
        # settings['s_folder_path'] = '/mnt/common/Jan/Programming/first_CNN_on_Radolan/dwd_nc/own_test_data'
        # settings['s_folder_path'] = '/mnt/common/Jan/Programming/first_CNN_on_Radolan/dwd_nc/own_test_data'
        settings['s_folder_path'] = 'dwd_nc/own_test_data'

        # settings['s_data_file_names'] = ['DE1200_RV_Recalc_20190101.nc']
        # settings['s_data_file_names'] = ['RV_recalc_data_2019-01.nc']
        settings['s_data_file_names'] = ['RV_recalc_data_2019-01_subset.nc']
        # settings['s_data_file_names'] = ['RV_recalc_data_2019-01_subset_bigger.nc']

        # settings['s_choose_time_span'] = True
        settings['s_choose_time_span'] = False
        # settings['s_time_span'] = (datetime.datetime(2019, 1, 1, 0), datetime.datetime(2019, 1, 1, 5))
        settings['s_time_span'] = (67, 150)  # <-- now done according to index (isel instead of sel)
        settings['s_upscale_c_to'] = 32  # 8
        settings['s_batch_size'] = 2
        settings['s_data_loader_chunk_size'] = 2
        settings['s_testing'] = True  # Runs tests at the beginning
        settings['s_min_rain_ratio_target'] = 0  # Deactivated # No Filter
        # FILTER NOT WORKING YET, ALWAYS RETURNS TRUE FOR TEST PURPOSES!!

    main(settings=settings, **settings)






