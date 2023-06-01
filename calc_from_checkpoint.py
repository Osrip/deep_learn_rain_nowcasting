from helper.checkpoint_handling import load_from_checkpoint, create_data_loaders
from load_data import inverse_normalize_data
from helper.helper_functions import one_hot_to_mm
from train_lightning import data_loading
from plotting.plot_images import plot_target_vs_pred_with_likelihood
from helper.helper_functions import load_zipped_pickle


def plot_images_inner(model, data_loader, filter_and_normalization_params, linspace_binning_params, ps_runs_path,
                      ps_run_name, ps_checkpoint_name, **__):

    filtered_indecies, mean_filtered_data, std_filtered_data, linspace_binning_min_unnormalized,\
        linspace_binning_max_unnormalized = filter_and_normalization_params

    linspace_binning_min, linspace_binning_max, linspace_binning = linspace_binning_params

    inv_norm = lambda x: inverse_normalize_data(x, mean_filtered_data, std_filtered_data, inverse_log=False,
                                                inverse_normalize=True)

    for i, (input_sequence, target_one_hot, target) in enumerate(data_loader):
        pred = model(input_sequence)

        pred_mm = one_hot_to_mm(pred, linspace_binning, linspace_binning_max, channel_dim=1, mean_bin_vals=True)

        if i == 0:
            plot_target_vs_pred_with_likelihood(inv_norm(target), inv_norm(pred_mm), pred,
                                                linspace_binning=inv_norm(linspace_binning),
                                                vmin=inv_norm(linspace_binning_min),
                                                vmax=inv_norm(linspace_binning_max),
                                                save_path_name= '{}/{}/plots/VAL_target_vs_pred_likelihood_{}.png'.format(ps_runs_path
                                                                                                        , ps_run_name
                                                                                                        , ps_checkpoint_name),
                                                title='Validation data (log, not normalized)')



def plot_images_outer(plot_settings, ps_runs_path, ps_run_name, ps_checkpoint_name, **__):

    filter_and_normalization_params = load_zipped_pickle('{}/{}/data/filter_and_normalization_params'.format(ps_runs_path, ps_run_name))
    linspace_binning_params = load_zipped_pickle('{}/{}/data/linspace_binning_params'.format(ps_runs_path, ps_run_name))

    model = load_from_checkpoint(ps_runs_path, ps_run_name, ps_checkpoint_name)
    train_data_loader, validation_data_loader = create_data_loaders(ps_runs_path, ps_run_name)
    plot_images_inner(model, train_data_loader, filter_and_normalization_params, linspace_binning_params, **plot_settings)
    plot_images_inner(model, validation_data_loader, filter_and_normalization_params, linspace_binning_params, **plot_settings)


if __name__ == '__main__':


    plot_settings = {
        'ps_runs_path': '/home/jan/jan/programming/first_CNN_on_Radolan/runs',
        'ps_run_name': 'Run_20230601-181344_test_profiler',
        'ps_checkpoint_name': 'model_epoch=1_val_loss=3.94.ckpt',
    }

    plot_images_outer(plot_settings, **plot_settings)




    #pass