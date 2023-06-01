from train_lightning import Network_l
from helper.helper_functions import load_zipped_pickle

def load_from_checkpoint(runs_path, run_name, checkpoint_name):
    checkpoint_path = '{}/{}/model/{}'.format(runs_path, run_name, checkpoint_name)
    settings = load_zipped_pickle('{}/{}/settings'.format(runs_path, run_name))
    linspace_binning_params = load_zipped_pickle('{}/{}/linspace_binning_params'.format(runs_path, run_name))

    model = Network_l.load_from_checkpoint(checkpoint_path=checkpoint_path, linspace_binning_params=linspace_binning_params,
                                           **settings)

    return model