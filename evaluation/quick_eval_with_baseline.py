from helper.checkpoint_handling import load_from_checkpoint, get_checkpoint_names

def ckpt_quick_eval_with_baseline(
        sample_coords,
        radolan_statistics_dict,
        linspace_binning_params,
        splits_to_predict_on,

        ckp_settings,  # Make sure to pass the settings of the checkpoint
        s_dirs,

        max_num_frames_per_split=None,

        **__,
):

    model = load_from_checkpoint(
        save_dir,
        checkpoint_name_to_predict,

        ckp_settings,
        **ckp_settings,
    )