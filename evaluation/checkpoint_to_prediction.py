from load_data_xarray import create_patches

def ckp_to_pred(
        settings,
        s_width_height_target,
        s_width_height,
        **__,
):

    # Define constants for pre-processing
    y_target, x_target = s_width_height_target, s_width_height_target  # 73, 137 # how many pixels in y and x direction
    y_input, x_input = s_width_height, s_width_height
    y_input_padding, x_input_padding = 0, 0  # No augmentation, thus no padding for evaluation

    (
        patches,
        # patches: xr.Dataset Patch dimensions y_outer, x_outer give one coordinate pair for each patch,
        # y_inner, x_inner give pixel dimensions for each patch
        data,
        # data: The unpatched data that has global pixel coordinates,
        data_shortened,
        # data_shortened: same as data, but beginning is missing (lead_time + num input frames) such that we can go
        # 'back in time' to go fram target time to input time.
    ) = create_patches(
        y_target,
        x_target,
        **settings
    )