from train_lightning import Network_l
from helper.helper_functions import load_zipped_pickle
from torch.utils.data import DataLoader


def load_from_checkpoint(runs_path, run_name, checkpoint_name):
    checkpoint_path = '{}/{}/model/{}'.format(runs_path, run_name, checkpoint_name)
    settings = load_zipped_pickle('{}/{}/data/settings'.format(runs_path, run_name))
    linspace_binning_params = load_zipped_pickle('{}/{}/data/linspace_binning_params'.format(runs_path, run_name))

    model = Network_l.load_from_checkpoint(checkpoint_path=checkpoint_path, linspace_binning_params=linspace_binning_params,
                                           **settings)

    return model


def create_data_loaders(runs_path, run_name):
    train_data_set = load_zipped_pickle('{}/{}/data/train_data_set'.format(runs_path, run_name))
    validation_data_set = load_zipped_pickle('{}/{}/data/validation_data_set'.format(runs_path, run_name))
    settings = load_zipped_pickle('{}/{}/data/settings'.format(runs_path, run_name))


    train_data_loader = DataLoader(train_data_set, batch_size=settings['s_batch_size'], shuffle=True, drop_last=True,
                                   num_workers=['s_num_workers_data_loader'])

    validation_data_loader = DataLoader(validation_data_set, batch_size=settings['s_batch_size'], shuffle=True, drop_last=True,
                                   num_workers=['s_num_workers_data_loader'])

    return train_data_loader, validation_data_loader