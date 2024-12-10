from evaluation.checkpoint_to_prediction import ckpt_to_pred
from evaluation.quick_eval_with_baseline import ckpt_quick_eval_with_baseline
from helper.checkpoint_handling import get_checkpoint_names, load_from_checkpoint
from plotting.plotting_pipeline import plot_logs_pipeline


def evaluation_pipeline(data_set_vars, ckpt_settings, plotting=False):
    '''
    This Pipeline is executed right after training or in 'plotting_only' mode
    '''


    (train_data_loader, validation_data_loader,
     training_steps_per_epoch, validation_steps_per_epoch,
     train_time_keys, val_time_keys, test_time_keys,
     train_sample_coords, val_sample_coords,
     radolan_statistics_dict,
     linspace_binning_params,) = data_set_vars



    # --- Get model checkpoint ---

    save_dir = ckpt_settings['s_dirs']['save_dir']

    checkpoint_names = get_checkpoint_names(save_dir)

    # Only do prediction for last checkpoint
    # TODO Make this best checkpoint on validation loss
    checkpoint_name = [name for name in checkpoint_names if 'last' in name][0]

    model = load_from_checkpoint(
        save_dir,
        checkpoint_name,

        ckpt_settings,
        **ckpt_settings,
    )

    # --- Plot logs ---
    if plotting:
        plot_logs_pipeline(
            training_steps_per_epoch,
            model,
            ckpt_settings, **ckpt_settings
        )

    # --- Quick evaluation and comparison to baseline over data set ---

    ckpt_quick_eval_with_baseline(
        model,
        checkpoint_name,
        val_sample_coords,
        radolan_statistics_dict,
        linspace_binning_params,

        ckpt_settings,
        **ckpt_settings
    )

    # --- Generate predictions that are saved to a zarr ---

    ckpt_to_pred(
        model,
        checkpoint_name,
        train_time_keys, val_time_keys, test_time_keys,
        radolan_statistics_dict,
        linspace_binning_params,
        max_num_frames_per_split=15,

        splits_to_predict_on=['val'],
        ckp_settings=ckpt_settings,
        **ckpt_settings,
    )
