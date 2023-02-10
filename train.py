import datetime

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from helper_functions import create_dilation_list
from modules_blocks import Network
import datetime
from load_data import load_data_sequence


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


def train(start_date_time: datetime.datetime, folder_path: str, future_iterations_from_start: int,
          minutes_per_iteration: int, width_height: int, learning_rate: int):
    accuracies = []
    sim_name = 'Run_·{}'.format(datetime.time.strftime("%Y%m%d-%H%M%S"))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    max_accuracy = 0
    data_sequence = load_data_sequence(
        start_date_time, folder_path, future_iterations_from_start, minutes_per_iteration, width_height)
    # TODO: Continue this!!




if __name__ == '__main__':
    model = Network(c_in=8, width_height_in=256)
    # Throws an error on remote venv for some reason
    optimized_model = check_backward(model, learning_rate=0.001, device='cpu')
    pass


