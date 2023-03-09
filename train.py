import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from helper_functions import create_dilation_list
from modules_blocks import Network
import datetime
from load_data import img_one_hot, PrecipitationDataset, load_data_sequence_preliminary
from torch.utils.data import Dataset, DataLoader
import numpy as np
from helper_functions import load_zipped_pickle, save_zipped_pickle
import os
from plotting.plot_img_histogram import plot_img_histogram
import copy
# from pysteps import verification
# fss = verification.get_method("FSS")


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
        for i, (input_sequence, target, _, _) in enumerate(validation_data_loader):
            input_sequence = input_sequence.float()
            input_sequence = input_sequence.to(device)
            target = T.CenterCrop(size=width_height_target)(target)
            target = target.float()
            target = target.to(device)

            pred = model(input_sequence)
            loss = criterion(pred, target)
            loss = float(loss.detach().cpu().numpy())
            validation_losses.append(loss)
    return np.mean(validation_losses)


def plot_losses(losses, validation_losses, plot_dir):
    plt.figure()
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # plt.savefig('{}/{}_loss.png'.format(plot_dir, sim_name), dpi=100)
    # plt.show()
    plt.plot(validation_losses, label='Validation Loss')
    plt.legend()
    plt.savefig('{}/{}_loss.png'.format(plot_dir, sim_name), dpi=100)
    plt.show()


def train(model, sim_name, device, learning_rate: int, num_epochs: int, num_input_time_steps: int,
          num_bins_crossentropy, width_height_target, batch_size, ratio_training_data,
          dirs, local_machine_mode, settings, **__):
    accuracies = []
    # Add one to
    num_pictures_loaded = num_input_time_steps + 1

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # TODO: Enable saving to pickle at some point
    print('Load Data', flush=True)
    data_sequence = load_data_sequence_preliminary(**settings)
    num_training_samples = int(np.shape(data_sequence)[0] * ratio_training_data)
    train_data_sequence = data_sequence[0:num_training_samples, :, :]
    validation_data_sequence = data_sequence[num_training_samples:, :, :]

    train_data_set = PrecipitationDataset(train_data_sequence, num_pictures_loaded, num_bins_crossentropy)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    del train_data_set

    validation_data_set = PrecipitationDataset(validation_data_sequence, num_pictures_loaded, num_bins_crossentropy)
    validation_data_loader = DataLoader(validation_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    del validation_data_set

    losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        inner_losses = []
        for i, (input_sequence, target, input_sequence_unnormalized, target_unnormalized) in enumerate(train_data_loader):
            input_sequence = input_sequence.float()

            input_sequence = input_sequence.to(device)
            input_sequence = input_sequence.to(device=device, dtype=torch.float32)
            target = target.to(device)
            input_sequence_unnormalized = input_sequence_unnormalized.to(device)
            target_unnormalized = target_unnormalized.to(device)

            print('Batch: {}'.format(i), flush=True)

            target = T.CenterCrop(size=width_height_target)(target)
            target = target.float()
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(input_sequence)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            inner_loss = float(loss.detach().cpu().numpy())
            # TODO: convert cudo tensor to numpy!! How to push to CPU for numpy conversion and then back to cuda??
            # loss_copy = copy.deepcopy(loss)
            # inner_loss = float(loss_copy.detach().to('cpu').numpy())
            # del loss_copy
            # loss = loss.to(device)
            inner_losses.append(inner_loss)
            # TODO: Currently Target is baseline for test purposes. Change that for obvious reasons!!!


        avg_inner_loss = np.mean(inner_losses)
        losses.append(avg_inner_loss)
        validation_loss = validate(model, validation_data_loader, **settings)
        validation_losses.append(validation_loss)
        print('Epoch: {} Training loss: {}, Validation loss: {}'.format(epoch, avg_inner_loss, validation_loss), flush=True)
        plot_losses(losses, validation_losses, dirs['plot_dir'])
        plot_img_histogram(pred, '{}/ep{}_pred_dist'.format(dirs['plot_dir'], epoch), title='Prediciton')
        plot_img_histogram(input_sequence_unnormalized, '{}/ep{}_input_dist'.format(dirs['plot_dir'], epoch), title='Input')
        plot_img_histogram(target_unnormalized, '{}/ep{}_target_dist'.format(dirs['plot_dir'], epoch),
                           title='Target')
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
    dirs['model_dir'] = '{}/model'.format(dirs['save_dir'])
    for _, make_dir in dirs.items():
        if not os.path.exists(make_dir):
            os.makedirs(make_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

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
            'learning_rate': 0.0001,  # Schedule this at some point??
            'num_epochs': 1000,
            'num_input_time_steps': 4,
            'optical_flow_input': False,  # Not yet working!
            'width_height_target': 32,
            'batch_size': 10,  # 10
            'save_trained_model': True,
            'load_model': False,
            'load_model_name': 'Run_·20230220-191041',
            'dirs': dirs,
            'device': device
        }

    if settings['local_machine_mode']:
        settings['data_variable_name'] = 'WN_forecast'
        settings['folder_path'] = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/first_CNN_on_Radolan/dwd_nc/test_data'
        settings['data_file_name'] = 'DE1200_RV_Recalc_20190101.nc'
        settings['choose_time_span'] = True
        settings['time_span'] = (datetime.datetime(2019, 1, 1, 0), datetime.datetime(2019, 1, 1, 5))
        settings['upscale_c_to'] = 16
        settings['batch_size'] = 10

    main(settings=settings, **settings)






