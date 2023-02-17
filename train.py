import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from helper_functions import create_dilation_list
from modules_blocks import Network
import datetime
from load_data import load_data_sequence, img_one_hot, PrecipitationDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np



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


def validate(model, validation_data_loader):
    criterion = nn.CrossEntropyLoss()
    validation_losses = []
    for i, (input_sequence, target) in enumerate(validation_data_loader):
        input_sequence = input_sequence.float()
        target = T.CenterCrop(size=width_height_target)(target)
        target = target.float()

        pred = model(input_sequence)
        loss = criterion(pred, target)
        loss = float(loss.detach().numpy())
        validation_losses.append(loss)
    return np.mean(validation_losses)


def plot_losses(losses, validation_losses, sim_name):
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Results/{}_loss.png'.format(sim_name), dpi=100)
    plt.show()
    plt.plot(validation_losses)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Results/{}_validation_loss.png'.format(sim_name), dpi=100)
    plt.show()


def train(model, train_start_date_time: datetime.datetime, device, folder_path: str, num_training_samples: int, num_validation_samples,
          minutes_per_iteration: int, width_height: int, learning_rate: int, num_epochs: int,
          num_input_time_steps: int, num_channels_one_hot_output, width_height_target, batch_size):
    accuracies = []
    # Add one to
    num_pictures_loaded = num_input_time_steps + 1

    sim_name = 'Run_·{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # TODO: Enable saving to pickle at some point
    data_sequence = load_data_sequence(train_start_date_time, folder_path, future_iterations_from_start=num_training_samples+num_validation_samples,
                                       width_height=width_height, minutes_per_iteration=minutes_per_iteration)
    train_data_sequence = data_sequence[0:num_training_samples, :, :]
    validation_data_sequence = data_sequence[num_training_samples:, :, :]

    train_data_set = PrecipitationDataset(train_data_sequence, num_pictures_loaded, num_channels_one_hot_output)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, drop_last=True)

    validation_data_set = PrecipitationDataset(validation_data_sequence, num_pictures_loaded, num_channels_one_hot_output)
    validation_data_loader = DataLoader(validation_data_set, batch_size=batch_size, shuffle=True, drop_last=True)

    losses = []
    validation_losses = []
    for epoch in range(num_epochs):
        inner_losses = []
        for i, (input_sequence, target) in enumerate(train_data_loader):

            print('Picture: {}'.format(i))
            input_sequence = input_sequence.float()
            target = T.CenterCrop(size=width_height_target)(target)
            target = target.float()

            optimizer.zero_grad()
            pred = model(input_sequence)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            inner_loss = float(loss.detach().numpy())
            inner_losses.append(inner_loss)
        avg_inner_loss = np.mean(inner_losses)
        losses.append(avg_inner_loss)
        validation_loss = validate(model, validation_data_loader)
        validation_losses.append(validation_loss)
        print('Epoch: {} Loss: {}'.format(epoch, avg_inner_loss))
        plot_losses(losses, validation_losses, sim_name)


# def train(model, start_date_time: datetime.datetime, device, folder_path: str, num_training_samples: int,
#           minutes_per_iteration: int, width_height: int, learning_rate: int, num_epochs: int,
#           num_input_time_steps: int, num_channels_one_hot_output, width_height_target):
#     accuracies = []
#     # Add one to
#     num_pictures_loaded = num_input_time_steps + 1
#
#     sim_name = 'Run_·{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     max_accuracy = 0
#     # TODO: Enable saving to pickle at some point
#     data_sequence = load_data_sequence(start_date_time, folder_path, future_iterations_from_start=num_training_samples,
#                                        width_height=width_height, minutes_per_iteration=minutes_per_iteration)
#     for epoch in range(num_epochs):
#         print('Epoch: {}'.format(epoch))
#         for i in range(np.shape(data_sequence)[0] - num_pictures_loaded):
#             print('Picture: {}'.format(i))
#             # TODO: IMPLEMENT BATCHES!!!
#             input_sequence = data_sequence[i:i+num_pictures_loaded-1, :, :]
#             input_sequence = torch.from_numpy(input_sequence)
#             input_sequence = input_sequence.float()
#             # Add dummy batch dimension
#             input_sequence = torch.unsqueeze(input_sequence, 0)
#             target = np.squeeze(data_sequence[i+num_pictures_loaded, :, :])  # Get rid of 1st dimension with np squeeze??
#             target = img_one_hot(target, num_channels_one_hot_output)
#             target = T.CenterCrop(size=width_height_target)(target)
#             target = torch.unsqueeze(target, 0)
#             target = target.float()
#             optimizer.zero_grad()
#
#             pred = model(input_sequence)
#             loss = criterion(pred, target)
#             loss.backward()
#             optimizer.step()

    # Todo: implement validation to estimate accuracy!






    # TODO: Continue this!!


if __name__ == '__main__':

    num_training_samples = 200 # Number of loaded pictures (first pics not used for training but only input)
    num_validation_samples = 200
    minutes_per_iteration = 5
    width_height = 256
    learning_rate = 0.0001
    num_epochs = 10000
    num_input_time_steps = 4
    optical_flow_input = False  # Not yet working!
    num_channels_one_hot_output = 32  # TODO: Check this!! Not 64??
    width_height_target = 32
    batch_size = 10

    model = Network(c_in=num_input_time_steps, width_height_in=width_height)
    # NETWORK STILL NEEDS NUMBER OF OUTPUT CHANNELS num_channels_one_hot_output !!!

    # Throws an error on remote venv for some reason
    # optimized_model = check_backward(model, learning_rate=0.001, device='cpu')
    train_start_date_time = datetime.datetime(2020, 12, 1)
    folder_path = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/weather_data/dwd_datensatz_bits/rv_recalc/RV_RECALC/hdf/'
    device = 'cpu'

    train(model, train_start_date_time, device, folder_path, num_training_samples, num_validation_samples, minutes_per_iteration, width_height,
          learning_rate, num_epochs, num_input_time_steps, num_channels_one_hot_output, width_height_target, batch_size)
    pass


