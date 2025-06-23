# Fix typing issues before any imports to be able to debug on 5090 machine
import os

from training_utils import data_loading, train_wrapper

os.environ["TORCH_DYNAMO_DISABLE"] = "1"

# Import and apply our typing fix
from helper.type_fix import apply_typing_fixes
apply_typing_fixes()

print("PyTorch dynamo disabled and typing fixes applied for debugging")


import torch
import datetime
import argparse

from helper import load_settings, load_zipped_pickle, no_special_characters
from plotting import plot_logs_pipeline
from tests import test_all
from evaluation import evaluation_pipeline


def create_s_dirs(sim_name, s_mode, s_save_dir, s_prediction_dir, **__):
    s_dirs = {}
    s_dirs['save_dir'] = os.path.join(s_save_dir, sim_name)
    s_dirs['prediction_dir'] = os.path.join(s_prediction_dir, sim_name)

    # s_dirs['save_dir'] = 'runs/{}'.format(s_sim_name)
    s_dirs['plot_dir']          = '{}/plots'.format(s_dirs['save_dir'])
    s_dirs['plot_dir_images']   = '{}/images'.format(s_dirs['plot_dir'])
    s_dirs['plot_dir_fss']      = '{}/fss'.format(s_dirs['plot_dir'])
    s_dirs['model_dir']         = '{}/model'.format(s_dirs['save_dir'])
    s_dirs['code_dir']          = '{}/code'.format(s_dirs['save_dir'])
    s_dirs['profile_dir']       = '{}/profile'.format(s_dirs['save_dir'])
    s_dirs['logs']              = '{}/logs'.format(s_dirs['save_dir'])
    s_dirs['data_dir']          = '{}/data'.format(s_dirs['save_dir'])
    s_dirs['batches_outputs']   = '{}/batches_outputs'.format(s_dirs['save_dir'])

    return s_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cluster',
                        choices=['cluster', 'local', 'debug'],
                        help="Mode: cluster, local, or debug")
    args = parser.parse_args()

    # Load settings based on mode
    settings = load_settings(args.mode)

    # Process simulation name suffix
    s_sim_name_suffix = settings.get('s_sim_name_suffix', 'dlbd_training_one_month')
    s_sim_name_suffix = no_special_characters(s_sim_name_suffix)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build simulation name based on mode.
    if args.mode in ['local', 'debug']:
        s_sim_name = 'Run_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        s_sim_name = 'Run_{}_ID_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                           int(os.environ.get('SLURM_JOB_ID', 0)))
    s_sim_name += s_sim_name_suffix

    s_dirs = {
        'save_dir': os.path.join(settings.get('s_save_dir'), s_sim_name),
        'prediction_dir': os.path.join(settings.get('s_prediction_dir'), s_sim_name)
    }

    # Append additional directory entries.
    s_dirs['plot_dir'] = f"{s_dirs['save_dir']}/plots"
    s_dirs['plot_dir_images'] = f"{s_dirs['plot_dir']}/images"
    s_dirs['plot_dir_fss'] = f"{s_dirs['plot_dir']}/fss"
    s_dirs['model_dir'] = f"{s_dirs['save_dir']}/model"
    s_dirs['code_dir'] = f"{s_dirs['save_dir']}/code"
    s_dirs['profile_dir'] = f"{s_dirs['save_dir']}/profile"
    s_dirs['logs'] = f"{s_dirs['save_dir']}/logs"
    s_dirs['data_dir'] = f"{s_dirs['save_dir']}/data"
    s_dirs['batches_outputs'] = f"{s_dirs['save_dir']}/batches_outputs"

    # Append extra keys to settings.
    settings['s_mode'] = args.mode
    settings['s_dirs'] = s_dirs
    settings['device'] = device
    settings['s_sim_name'] = s_sim_name


    if not settings['s_plotting_only']:
        for _, make_dir in s_dirs.items():
            if not os.path.exists(make_dir):
                os.makedirs(make_dir)

    if settings['s_no_plotting']:
        for en in ['s_plot_average_preds_boo', 's_plot_pixelwise_preds_boo', 's_plot_target_vs_pred_boo',
                   's_plot_mse_boo', 's_plot_losses_boo', 's_plot_img_histogram_boo']:
            settings[en] = False

    if settings['s_testing']:
        test_all()

    if not settings['s_plotting_only']:
        # --- Normal training ---
        data_set_vars = data_loading(settings, **settings)

        (train_data_loader, validation_data_loader,
        training_steps_per_epoch, validation_steps_per_epoch,
        train_time_keys, val_time_keys, test_time_keys,
        train_sample_coords, val_sample_coords,
        radolan_statistics_dict,
        linspace_binning_params,) = data_set_vars

        model_l, training_steps_per_epoch, sigma_schedule_mapping = train_wrapper(
            *data_set_vars,
            settings,
            **settings
        )

        plot_logs_pipeline(
            training_steps_per_epoch,
            model_l,
            settings, **settings
        )

        evaluation_pipeline(data_set_vars, settings)

    else:
        # --- Plotting only ---
        load_dirs = create_s_dirs(settings['s_plot_sim_name'], **settings)
        training_steps_per_epoch = load_zipped_pickle('{}/training_steps_per_epoch'.format(load_dirs['data_dir']))
        sigma_schedule_mapping = load_zipped_pickle('{}/sigma_schedule_mapping'.format(load_dirs['data_dir']))
        ckpt_settings = load_zipped_pickle('{}/settings'.format(load_dirs['data_dir']))

        ckpt_settings['s_dirs']['save_dir'] = load_dirs['save_dir']

        # Convert some of the loaded settings to the current settings
        ckpt_settings['s_baseline_path']            = settings['s_baseline_path']
        ckpt_settings['s_baseline_variable_name']   = settings['s_baseline_variable_name']
        ckpt_settings['s_num_input_frames_baseline'] = settings['s_num_input_frames_baseline']

        ckpt_settings['s_num_gpus']                 = settings['s_num_gpus']
        ckpt_settings['s_baseline_path']            = settings['s_baseline_path']
        ckpt_settings['s_baseline_variable_name']   = settings['s_baseline_variable_name']
        ckpt_settings['s_num_input_frames_baseline']= settings['s_num_input_frames_baseline']

        # Settings related to evaluation:
        ckpt_settings['s_fss'] = settings['s_fss']
        ckpt_settings['s_fss_scales'] = settings['s_fss_scales']
        ckpt_settings['s_fss_thresholds'] = settings['s_fss_thresholds']

        ckpt_settings['s_dlbd_eval'] = settings['s_dlbd_eval']
        ckpt_settings['s_sigmas_dlbd_eval'] = settings['s_sigmas_dlbd_eval']

        # Pass settings of the loaded run to get the according data_set_vars
        data_set_vars = data_loading(ckpt_settings, **ckpt_settings)

        evaluation_pipeline(data_set_vars, ckpt_settings)


if __name__ == '__main__':
    main()
















