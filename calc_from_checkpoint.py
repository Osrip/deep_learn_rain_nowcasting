import torch
import pytorch_lightning as pl
import numpy as np
from train_lightning import Network_l
from helper_functions import load_zipped_pickle

def load_from_checkpoint(runs_path, run_name, checkpoint_name):
    checkpoint_path = '{}/{}/model/{}'.format(runs_path, run_name, checkpoint_name)
    bla = load_zipped_pickle('{}/{}/Network_l_class'.format(runs_path, run_name))
    settings = load_zipped_pickle('{}/{}/settings'.format(runs_path, run_name))
    linspace_binning_params = load_zipped_pickle('{}/{}/linspace_binning_params'.format(runs_path, run_name))

    model = Network_l.load_from_checkpoint(checkpoint_path=checkpoint_path, linspace_binning_params=linspace_binning_params,
                                           **settings)


    return model


if __name__ == '__main__':


    plot_settings = {
        'ps_runs_path': '/home/jan/jan/programming/first_CNN_on_Radolan/runs',
        'ps_run_name': 'Run_20230601-124534_test_profiler',
        'ps_checkpoint_name': 'model_epoch=0_val_loss=4.12.ckpt',
    }

    model = load_from_checkpoint(plot_settings['ps_runs_path'], plot_settings['ps_run_name'],
                                 plot_settings['ps_checkpoint_name'])



    #pass