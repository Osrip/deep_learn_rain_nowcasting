import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from helper_functions import create_dilation_list
from modules_blocks import Network


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


if __name__ == '__main__':
    model = Network(c_in=8, width_height_in=256)
    optimized_model = check_backward(model, learning_rate=0.001, device='cpu')
    pass


