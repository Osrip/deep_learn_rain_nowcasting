import datetime

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from helper_functions import create_dilation_list
from modules_blocks import Network
import datetime
from load_data import load_data_sequence, img_one_hot
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


def train(model, start_date_time: datetime.datetime, device, folder_path: str, num_training_samples: int,
          minutes_per_iteration: int, width_height: int, learning_rate: int, num_epochs: int,
          num_input_time_steps: int, num_channels_one_hot_output, width_height_target):
    accuracies = []
    # Add one to
    num_pictures_loaded = num_input_time_steps + 1

    sim_name = 'Run_·{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    max_accuracy = 0
    # TODO: Enable saving to pickle at some point
    data_sequence = load_data_sequence(start_date_time, folder_path, future_iterations_from_start=num_training_samples,
                                       width_height=width_height, minutes_per_iteration=minutes_per_iteration)
    for epoch in range(num_epochs):
        for i in range(np.shape(data_sequence)[0] - num_pictures_loaded):
            # TODO: IMPLEMENT BATCHES!!!
            input_sequence = data_sequence[i:i+num_pictures_loaded-1, :, :]
            input_sequence = torch.from_numpy(input_sequence)
            input_sequence = input_sequence.float()
            # Add dummy batch dimension
            input_sequence = torch.unsqueeze(input_sequence, 0)
            target = np.squeeze(data_sequence[i+num_pictures_loaded, :, :]) # Get rid of 1st dimension with np squeeze??
            target = img_one_hot(target, num_channels_one_hot_output)
            target = T.CenterCrop(size=width_height_target)(target)
            optimizer.zero_grad()

            pred = model(input_sequence)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
    # Todo: implement validation to estimate accuracy!






    # TODO: Continue this!!




if __name__ == '__main__':

    num_training_samples = 20 # Number of loaded pictures (first pics not used for training but only input)
    minutes_per_iteration = 5
    width_height = 256
    learning_rate = 0.0001
    num_epochs = 50
    num_input_time_steps = 4
    optical_flow_input = False  # Not yet working!
    num_channels_one_hot_output = 32  # TODO: Check this!! Not 64??
    width_height_target = 32

    model = Network(c_in=num_input_time_steps, width_height_in=width_height)
    # NETWORK STILL NEEDS NUMBER OF OUTPUT CHANNELS num_channels_one_hot_output !!!

    # Throws an error on remote venv for some reason
    # optimized_model = check_backward(model, learning_rate=0.001, device='cpu')
    start_date_time = datetime.datetime(2020, 12, 1)
    folder_path = '/media/jan/54093204402DAFBA/Jan/Programming/Butz_AG/weather_data/dwd_datensatz_bits/rv_recalc/RV_RECALC/hdf/'
    device = 'cpu'

    train(model, start_date_time, device, folder_path, num_training_samples, minutes_per_iteration, width_height,
          learning_rate, num_epochs, num_input_time_steps, num_channels_one_hot_output, width_height_target)
    pass


