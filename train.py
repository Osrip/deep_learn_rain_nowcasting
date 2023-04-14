import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from helper_functions import create_dilation_list
from modules_blocks import Network
import datetime
from load_data import img_one_hot, PrecipitationFilteredDataset, load_data_sequence_preliminary, normalize_data,\
    inverse_normalize_data, filtering_data_scraper, lognormalize_data
from torch.utils.data import Dataset, DataLoader

import numpy as np
from helper_functions import load_zipped_pickle, save_zipped_pickle, one_hot_to_mm, save_settings, save_whole_project
import os
from plotting.plot_img_histogram import plot_img_histogram
from plotting.plot_images import plot_image
from plotting.plot_quality_metrics import plot_mse, plot_losses, plot_average_preds, plot_pixelwise_preds
import warnings
from tests.test_basic_functions import test_all
from hurry.filesize import size
from tqdm import tqdm

import copy
# from pysteps import verification
# fss = verification.get_method("FSS")

mse_loss = torch.nn.MSELoss()


def check_backward(model, learning_rate, device):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    input = torch.randn((4, 8, 256, 256), device=device)
    target = torch.randn((4, 64, 32, 32), device=device)
    pred = model(input)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    return model


def print_gpu_memory():
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print("Memory: Total: {}, Free: {}, Used:{}".format(size(info.total), size(info.free), size(info.used)))

def validate(model, validation_data_loader, width_height_target, **__):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        validation_losses = []
        for i, (input_sequence, target_one_hot, target) in enumerate(validation_data_loader):
            input_sequence = input_sequence.float()
            input_sequence = input_sequence.to(device)
            target_one_hot = T.CenterCrop(size=width_height_target)(target_one_hot)
            target_one_hot = target_one_hot.float()
            target_one_hot = target_one_hot.to(device)

            pred = model(input_sequence)
            loss = criterion(pred, target_one_hot)
            loss = float(loss.detach().cpu().numpy())
            validation_losses.append(loss)
    return validation_losses


def train(model, sim_name, device, learning_rate: int, num_epochs: int, num_input_time_steps: int, num_lead_time_steps,
          num_bins_crossentropy, width_height_target, batch_size, ratio_training_data,
          dirs, local_machine_mode, log_transform, normalize, plot_average_preds_boo, plot_pixelwise_preds_boo, settings, **__):


    if log_transform:
        transform_f = lambda x: np.log(x + 1)
    else:
        transform_f = lambda x: x

    save_settings(settings, dirs['save_dir'])

    # try:
    save_whole_project(dirs['code_dir'])
    # except Exception:
    #     print('Could not create backup copy of code')
    # num_pictures_loaded = num_input_time_steps + 1 + num_lead_time_steps

    # relative index of last input picture (starting from first input picture as idx 1)
    last_input_rel_idx = num_input_time_steps
    #  relative index of target picture (starting from first input picture as idx 1)
    target_rel_idx = num_input_time_steps + 1 + num_lead_time_steps
    # Number of pictures

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # TODO: Enable saving to pickle at some point
    print('Load Data', flush=True)

    filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized, linspace_binning_max_unnormalized =\
        filtering_data_scraper(transform_f=transform_f, last_input_rel_idx=last_input_rel_idx, target_rel_idx=target_rel_idx,
                               **settings)

    # Normalize linspace binning thresholds now that data is available
    linspace_binning_min = lognormalize_data(linspace_binning_min_unnormalized, mean_filtered_data, std_filtered_data,
                                             transform_f, normalize) - 0.00001
    # Subtract a small number to account for rounding errors made in the normalization process
    linspace_binning_max = lognormalize_data(linspace_binning_max_unnormalized, mean_filtered_data, std_filtered_data,
                                             transform_f, normalize)

    linspace_binning_min = linspace_binning_min - 0.1
    linspace_binning_max = linspace_binning_max + 0.1

    linspace_binning = np.linspace(linspace_binning_min, linspace_binning_max, num=num_bins_crossentropy,
                                   endpoint=False)  # num_indecies + 1 as the very last entry will never be used

    num_training_samples = int(len(filtered_indecies) * ratio_training_data)
    train_filtered_indecies = filtered_indecies[0:num_training_samples]
    validation_filtered_indecies = filtered_indecies[num_training_samples:]

    train_data_set = PrecipitationFilteredDataset(train_filtered_indecies, mean_filtered_data, std_filtered_data,
                                                  linspace_binning_min, linspace_binning_max, linspace_binning, transform_f, **settings)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    del train_data_set

    validation_data_set = PrecipitationFilteredDataset(validation_filtered_indecies, mean_filtered_data,
                                                       std_filtered_data, linspace_binning_min, linspace_binning_max,
                                                       linspace_binning, transform_f, **settings)

    validation_data_loader = DataLoader(validation_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    del validation_data_set

    print('{} batches'.format(len(train_data_loader)))
    losses = []
    validation_losses = []
    relative_mses = []
    presistence_target_mses = []
    model_target_mses = []

    if plot_average_preds_boo or plot_pixelwise_preds_boo:
        all_pred_mm = []
        all_target_mm = []

    for epoch in range(num_epochs):
        if device.type == 'cuda':
            print_gpu_memory()

        inner_losses = []
        inner_relative_mses = []
        inner_presistence_target_mses = []
        inner_model_target_mses = []
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



            target = T.CenterCrop(size=width_height_target)(target)
            target_one_hot = T.CenterCrop(size=width_height_target)(target_one_hot)
            target_one_hot = target_one_hot.float()
            target_one_hot = target_one_hot.to(device)
            target = target.to(device).detach()

            optimizer.zero_grad()
            pred = model(input_sequence)
            loss = criterion(pred, target_one_hot)
            loss.backward()
            optimizer.step()

            # Quality metrics
            # TODO WHY ARE THERE NEGATIVE VALUES IN PRED MM??? --> CHECK NORMALIZATION
            pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)
            pred_mm = torch.from_numpy(pred_mm).to(device).detach()


            if plot_average_preds_boo or plot_pixelwise_preds_boo:
                for i in range(pred_mm.shape[0]):

                    pred_mm_arr = pred_mm.detach().cpu().numpy()
                    pred_mm_arr = inverse_normalize_data(pred_mm_arr, mean_filtered_data, std_filtered_data)
                    all_pred_mm.append(pred_mm_arr[i, :, :])

                    target_arr = target.detach().cpu().numpy()
                    target_arr = inverse_normalize_data(target_arr, mean_filtered_data, std_filtered_data)
                    all_target_mm.append(target_arr[i, :, :])

            inner_loss = float(loss.detach().cpu().numpy())
            # TODO: convert cudo tensor to numpy!! How to push to CPU for numpy conversion and then back to cuda??
            # loss_copy = copy.deepcopy(loss)
            # inner_loss = float(loss_copy.detach().to('cpu').numpy())
            # del loss_copy
            # loss = loss.to(device)
            inner_losses.append(inner_loss)

            persistence = input_sequence[:, -1, :, :]
            persistence = T.CenterCrop(size=width_height_target)(persistence)
            mse_persistence_target = mse_loss(persistence, target).item()
            mse_model_target = mse_loss(pred_mm, target).item()
            relative_mse = 1-mse_persistence_target/mse_model_target
            inner_relative_mses.append(relative_mse)
            inner_presistence_target_mses.append(mse_persistence_target)
            inner_model_target_mses.append(mse_model_target)


            # TODO: Currently Target is baseline for test purposes. Change that for obvious reasons!!!
            if i == 0:

                inv_norm = lambda x: inverse_normalize_data(x, mean_filtered_data, std_filtered_data, inverse_log=False,
                                                            inverse_normalize=True)

                plot_image(inv_norm(target[0, :, :]), vmin=inv_norm(linspace_binning_min), vmax=inv_norm(linspace_binning_max),
                           save_path_name='{}/ep{}_target'.format(dirs['plot_dir_images'], epoch),
                           title='Target')

                plot_image(inv_norm(pred_mm[0, :, :]), vmin=inv_norm(linspace_binning_min), vmax=inv_norm(linspace_binning_max),
                           save_path_name='{}/ep{}_pred'.format(dirs['plot_dir_images'], epoch),
                           title='Prediction')

        relative_mses.append(inner_relative_mses)
        presistence_target_mses.append(inner_presistence_target_mses)
        model_target_mses.append(inner_model_target_mses)
        losses.append(inner_losses)




        avg_inner_loss = np.mean(inner_losses)

        inner_validation_losses = validate(model, validation_data_loader, **settings)
        validation_losses.append(inner_validation_losses)
        avg_inner_validation_loss = np.mean(inner_validation_losses)
        print('Epoch: {} Training loss: {}, Validation loss: {}'.format(epoch, avg_inner_loss, avg_inner_validation_loss),
              flush=True)
        plot_mse([relative_mses], label_list=['relative MSE'],
                 save_path_name='{}/ep{}_relative_mse'.format(dirs['plot_dir'], epoch), title='Relative MSE')
        plot_mse([presistence_target_mses, model_target_mses], label_list=['Persistence Target MSE', 'Model Target MSE'],
                 save_path_name='{}/ep{}_mse'.format(dirs['plot_dir'], epoch),
                 title='MSE')
        plot_losses(losses, validation_losses, save_path_name='{}/{}_loss.png'.format(dirs['plot_dir'], sim_name))
        plot_img_histogram(pred_mm, '{}/ep{}_pred_dist'.format(dirs['plot_dir'], epoch), linspace_binning_min,
                           linspace_binning_max, ignore_min_max=False, title='Prediciton', **settings)
        plot_img_histogram(input_sequence, '{}/ep{}_input_dist'.format(dirs['plot_dir'], epoch),
                           linspace_binning_min, linspace_binning_max, title='Input', **settings)
        plot_img_histogram(target, '{}/ep{}_target_dist'.format(dirs['plot_dir'], epoch),
                           linspace_binning_min, linspace_binning_max, ignore_min_max=False, title='Target', **settings)

        if plot_average_preds_boo:
            plot_average_preds(all_pred_mm, all_target_mm, '{}/average_preds'.format(dirs['plot_dir']))

        if plot_pixelwise_preds_boo:
            plot_pixelwise_preds(all_pred_mm, all_target_mm, '{}/pixelwise_preds'.format(dirs['plot_dir']))


    return model


def main(save_trained_model, load_model, num_input_time_steps, upscale_c_to, load_model_name, width_height,
         num_bins_crossentropy, testing, settings, **_):
    if testing:
        test_all()

    if load_model:
        model = load_zipped_pickle('runs/{}/model/trained_model'.format(load_model_name))
    else:
        model = Network(c_in=num_input_time_steps, upscale_c_to=upscale_c_to,
                        num_bins_crossentropy=num_bins_crossentropy, width_height_in=width_height)
    model = model.to(device)
    # NETWORK STILL NEEDS NUMBER OF OUTPUT CHANNELS num_channels_one_hot_output !!!

    # Throws an error on remote venv for some reason
    # optimized_model = check_backward(model, learning_rate=0.001, device='cpu')



    print('Training started on {}'.format(device), flush=True)

    trained_model = train(model=model, settings=settings, **settings)
    if save_trained_model:
        save_zipped_pickle('{}/trained_model'.format(dirs['model_dir']), trained_model)


if __name__ == '__main__':
    #  Training data
    # num_training_samples = 20  # 1000  # Number of loaded pictures (first pics not used for training but only input)
    # num_validation_samples = 20  # 600

    # train_start_date_time = datetime.datetime(2020, 12, 1)
    # folder_path = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/weather_data/dwd_datensatz_bits/rv_recalc/RV_RECALC/hdf/'

    local_machine_mode = True

    sim_name_suffix = ''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        import nvidia_smi
        nvidia_smi.nvmlInit()
    # device = 'cpu'

    if local_machine_mode:
        sim_name = 'Run_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        sim_name = 'Run_{}_ID_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), int(os.environ['SLURM_JOB_ID'])) # SLURM_ARRAY_TASK_ID

    dirs = {}
    dirs['save_dir'] = 'runs/{}{}'.format(sim_name, sim_name_suffix)
    dirs['plot_dir'] = '{}/plots'.format(dirs['save_dir'])
    dirs['plot_dir_images'] = '{}/images'.format(dirs['plot_dir'])
    dirs['model_dir'] = '{}/model'.format(dirs['save_dir'])
    dirs['code_dir'] = '{}/code'.format(dirs['save_dir'])

    for _, make_dir in dirs.items():
        if not os.path.exists(make_dir):
            os.makedirs(make_dir)

    settings =\
        {
            'local_machine_mode': local_machine_mode,
            'sim_name': sim_name,
            'sim_same_suffix': sim_name_suffix,

            'folder_path': '/mnt/qb/butz/bst981/weather_data/dwd_nc/rv_recalc_months/rv_recalc_months',
            'data_file_names': ['RV_recalc_data_2019-01.nc'],  # ['RV_recalc_data_2019-0{}.nc'.format(i+1) for i in range(9)],
            'data_variable_name': 'RV_recalc',
            'choose_time_span': False,
            'time_span': (datetime.datetime(2020, 12, 1), datetime.datetime(2020, 12, 1)),
            'ratio_training_data': 0.6,

            # Parameters that give the network architecture
            'upscale_c_to': 64, #128, # 512,
            'num_bins_crossentropy': 32,

            # 'minutes_per_iteration': 5,
            'width_height': 256,
            'width_height_target': 32,
            'learning_rate': 0.0001,  # Schedule this at some point??
            'num_epochs': 1000,
            'num_input_time_steps': 4, # The number of subsequent time steps that are used for one predicition
            'num_lead_time_steps': 5, # The number of pictures that are skipped from last input time step to target, starts with 0
            'optical_flow_input': False,  # Not yet working!
            'batch_size': 26,  # batch size 22: Total: 32G, Free: 6G, Used:25G | Batch size 26: Total: 32G, Free: 1G, Used:30G --> vielfache von 8 am besten
            'save_trained_model': True,
            'load_model': False,
            'load_model_name': 'Run_·20230220-191041',
            'dirs': dirs,
            'device': device,

            # Log transform input/ validation data --> log binning --> log(x+1)
            'log_transform': True,
            'normalize': True,

            'min_rain_ratio_target': 0.01, #Deactivated  # The minimal amount of rain required in the 32 x 32 target for target and its
            # prior input sequence to make it through the filter into the training data

            'testing': True,

            # Plotting stuff
            'plot_average_preds_boo': True,
            'plot_pixelwise_preds_boo': True

        }







    if settings['local_machine_mode']:
        # settings['data_variable_name'] = 'WN_forecast'
        settings['data_variable_name'] = 'RV_recalc'

        # settings['folder_path'] = 'dwd_nc/test_data'
        # settings['folder_path'] = '/mnt/common/Jan/Programming/weather_data/dwd_nc/rv_recalc_months'
        # settings['folder_path'] = '/mnt/common/Jan/Programming/first_CNN_on_Radolan/dwd_nc/own_test_data'
        # settings['folder_path'] = '/mnt/common/Jan/Programming/first_CNN_on_Radolan/dwd_nc/own_test_data'
        settings['folder_path'] = 'dwd_nc/own_test_data'

        # settings['data_file_names'] = ['DE1200_RV_Recalc_20190101.nc']
        # settings['data_file_names'] = ['RV_recalc_data_2019-01.nc']
        settings['data_file_names'] = ['RV_recalc_data_2019-01_subset.nc']
        # settings['data_file_names'] = ['RV_recalc_data_2019-01_subset_bigger.nc']

        # settings['choose_time_span'] = True
        settings['choose_time_span'] = False
        # settings['time_span'] = (datetime.datetime(2019, 1, 1, 0), datetime.datetime(2019, 1, 1, 5))
        settings['time_span'] = (67, 150)  # <-- now done according to index (isel instead of sel)
        settings['upscale_c_to'] = 32  # 8
        settings['batch_size'] = 2
        settings['testing'] = True  # Runs tests at the beginning
        settings['min_rain_ratio_target'] = 0  # Deactivated # No Filter
        # FILTER NOT WORKING YET, ALWAYS RETURNS TRUE FOR TEST PURPOSES!!

    main(settings=settings, **settings)






