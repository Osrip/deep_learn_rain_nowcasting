import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from helper_functions import create_dilation_list
from modules_blocks import Network
import datetime
from load_data import img_one_hot, PrecipitationDataset, load_data_sequence_preliminary, normalize_data
from torch.utils.data import Dataset, DataLoader
import numpy as np
from helper_functions import load_zipped_pickle, save_zipped_pickle, one_hot_to_mm
import os
from plotting.plot_img_histogram import plot_img_histogram
from plotting.plot_images import plot_image
from plotting.plot_quality_metrics import plot_mse, plot_losses
import warnings

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



def train(model, sim_name, device, learning_rate: int, num_epochs: int, num_input_time_steps: int,
          num_bins_crossentropy, width_height_target, batch_size, ratio_training_data,
          dirs, local_machine_mode, log_transform, normalize, settings, **__):
    accuracies = []
    # Add one to
    num_pictures_loaded = num_input_time_steps + 1

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # TODO: Enable saving to pickle at some point
    print('Load Data', flush=True)

    data_sequence = load_data_sequence_preliminary(**settings)

    if log_transform:
        # Log transform with log x+1 to handle zeros
        data_sequence = np.log(data_sequence+1)
    if normalize:
        data_sequence, mean_data, std_data = normalize_data(data_sequence)
    else:
        mean_data = np.nan
        std_data = np.nan

    # min and max in log space if log transform True!
    linspace_binning_min = np.min(data_sequence)
    linspace_binning_max = np.max(data_sequence)

    num_training_samples = int(np.shape(data_sequence)[0] * ratio_training_data)
    train_data_sequence = data_sequence[0:num_training_samples, :, :]
    validation_data_sequence = data_sequence[num_training_samples:, :, :]



    train_data_set = PrecipitationDataset(train_data_sequence, num_pictures_loaded, num_bins_crossentropy
                                          , linspace_binning_min, linspace_binning_max)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    linspace_binning = train_data_set.linspace_binning
    del train_data_set

    validation_data_set = PrecipitationDataset(validation_data_sequence, num_pictures_loaded, num_bins_crossentropy,
                                               linspace_binning_min, linspace_binning_max)
    validation_data_loader = DataLoader(validation_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    if not (linspace_binning == validation_data_set.linspace_binning).all():
        warnings.warn('Different linspace binning applied in training and validation data set!')

    del validation_data_set

    losses = []
    validation_losses = []
    relative_mses = []
    presistence_target_mses = []
    model_target_mses = []

    for epoch in range(num_epochs):
        inner_losses = []
        inner_relative_mses = []
        inner_presistence_target_mses = []
        inner_model_target_mses = []
        for i, (input_sequence, target_one_hot, target) in enumerate(train_data_loader):
            input_sequence = input_sequence.float()

            input_sequence = input_sequence.to(device)
            input_sequence = input_sequence.to(device=device, dtype=torch.float32)
            target_one_hot = target_one_hot.to(device)


            print('Batch: {}'.format(i), flush=True)
            target = T.CenterCrop(size=width_height_target)(target)
            target_one_hot = T.CenterCrop(size=width_height_target)(target_one_hot)
            target_one_hot = target_one_hot.float()
            target_one_hot = target_one_hot.to(device)

            optimizer.zero_grad()
            pred = model(input_sequence)
            loss = criterion(pred, target_one_hot)
            loss.backward()
            optimizer.step()

            # Quality metrics
            pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)
            pred_mm = torch.from_numpy(pred_mm)
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

                plot_image(target[0, :, :], save_path_name='{}/ep{}_target'.format(dirs['plot_dir_images'], epoch),
                           title='Target')

                plot_image(pred_mm[0, :, :], save_path_name='{}/ep{}_pred'.format(dirs['plot_dir_images'], epoch),
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
        plot_mse([relative_mses], label_list=['relative MSE'], save_path_name='{}/ep{}_relative_mse'.format(dirs['plot_dir'], epoch),
                 title='Relative MSE')
        plot_mse([presistence_target_mses, model_target_mses], label_list=['Persistence Target MSE', 'Model Target MSE'],
                 save_path_name='{}/ep{}_mse'.format(dirs['plot_dir'], epoch),
                 title='MSE')
        plot_losses(losses, validation_losses, save_path_name='{}/{}_loss.png'.format(dirs['plot_dir'], sim_name))
        plot_img_histogram(pred, '{}/ep{}_pred_dist'.format(dirs['plot_dir'], epoch), linspace_binning_min,
                           linspace_binning_max, ignore_min_max=True, title='Prediciton', **settings)
        plot_img_histogram(input_sequence, '{}/ep{}_input_dist'.format(dirs['plot_dir'], epoch),
                           linspace_binning_min, linspace_binning_max, title='Input', **settings)
        plot_img_histogram(target, '{}/ep{}_target_dist'.format(dirs['plot_dir'], epoch),
                           linspace_binning_min, linspace_binning_max, title='Target', **settings)
    return model


def main(save_trained_model, load_model, num_input_time_steps, upscale_c_to, load_model_name, width_height,
         num_bins_crossentropy, settings, **_):
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

    sim_name = 'Run_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    dirs = {}
    dirs['save_dir'] = 'runs/{}'.format(sim_name)
    dirs['plot_dir'] = '{}/plots'.format(dirs['save_dir'])
    dirs['plot_dir_images'] = '{}/images'.format(dirs['plot_dir'])
    dirs['model_dir'] = '{}/model'.format(dirs['save_dir'])
    for _, make_dir in dirs.items():
        if not os.path.exists(make_dir):
            os.makedirs(make_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    settings =\
        {
            'local_machine_mode': True,

            'sim_name': sim_name,
            'folder_path': '/mnt/qb/butz/bst981/weather_data/dwd_nc/rv_recalc_months/rv_recalc_months',
            'data_file_name': 'RV_recalc_data_2019-01.nc',
            'data_variable_name': 'RV_recalc',
            'choose_time_span': False,
            'time_span': (datetime.datetime(2020, 12, 1), datetime.datetime(2020, 12, 1)),
            'ratio_training_data': 0.6,

            # Parameters that give the network architecture
            'upscale_c_to': 512,
            'num_bins_crossentropy': 64,

            # 'minutes_per_iteration': 5,
            'width_height': 256,
            'width_height_target': 32,
            'learning_rate': 0.0001,  # Schedule this at some point??
            'num_epochs': 1000,
            'num_input_time_steps': 4,
            'optical_flow_input': False,  # Not yet working!
            'batch_size': 10,  # 10
            'save_trained_model': True,
            'load_model': False,
            'load_model_name': 'Run_·20230220-191041',
            'dirs': dirs,
            'device': device,

            # Log transform input/ validation data --> log binning --> log(x+1)
            'log_transform': True,
            'normalize': True,
        }

    if settings['local_machine_mode']:
        settings['data_variable_name'] = 'WN_forecast'
        settings['folder_path'] = 'dwd_nc/test_data'
        settings['data_file_name'] = 'DE1200_RV_Recalc_20190101.nc'
        settings['choose_time_span'] = True
        settings['time_span'] = (datetime.datetime(2019, 1, 1, 0), datetime.datetime(2019, 1, 1, 5))
        settings['upscale_c_to'] = 8
        settings['batch_size'] = 2

    main(settings=settings, **settings)






