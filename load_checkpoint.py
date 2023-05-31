import torch
import pytorch_lightning as pl
import numpy as np
from train_lightning import Network_l

def load_from_checkpoint(checkpoint_path):
    model = Network_l.load_from_checkpoint(checkpoint_path)
    return model


if __name__ == '__main__':

    model = load_from_checkpoint('/home/jan/jan/programming/first_CNN_on_Radolan/runs/Run_20230526-185300_test_profiler/model/model_epoch=19_val_loss=3.88.ckpt')
    pass