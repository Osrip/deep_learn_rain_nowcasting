import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pysteps import io, nowcasts, rcparams
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.utils import conversion, dimension, transformation
from pysteps.visualization import plot_precip_field


def load_data(
        ensemble_num,
        lead_time_steps,
        seed,
        mins_per_time_step,
        radolan_path,
        **__
):
    dataset = xr.open_dataset(radolan_path, engine='zarr') # , chunks=None # , chunks=None according to Sebastian more efficient as it avoids dask (default is chunks=1)

    dataset = dataset.squeeze()

    # Set all negative values of the dataset to 0
    data_min = dataset.min(skipna=True, dim=None).RV_recalc.values
    if data_min < -0.1:
        raise ValueError(f'The min value of the dataset is {data_min}, which is below the arbitrary threshold of -0.1')

    dataset = dataset.where((dataset >= 0) | np.isnan(dataset), 0)
    # The where function keeps values where the condition is True and replaces the rest (where it's False)
    # with the value specified, in this case 0.
    print(f'min val in dataset is {dataset.min(skipna=True, dim=None).RV_recalc.values}')
    R = dataset['RV_recalc'].values

    return R, dataset


def predict(
        R,
        idx_slice: tuple,

        ensemble_num,
        lead_time_steps,
        seed,
        mins_per_time_step,
        radolan_path,
        num_input_frames,
        **__
):
    # --- Processing ---
    # DB transform
    R_normed = transformation.dB_transform(R, threshold=0.1, zerovalue=-15.0)[0]
    # This returns a tuple with metatdata, where first entry is actual data

    # Estimate the motion field
    V = dense_lucaskanade(R_normed[idx_slice[0]: idx_slice[1], :, :])
    # The STEPS nowcast
    nowcast_method = nowcasts.get_method("steps")
    R_normed_f = nowcast_method(
        R_normed[idx_slice[0]: idx_slice[1], :, :],
        V,
        lead_time_steps,
        ensemble_num,
        n_cascade_levels=6,
        R_thr=-10.0,
        kmperpixel=1.0,
        timestep=mins_per_time_step,
        noise_method="nonparametric",
        vel_pert_method="bps",
        mask_method="incremental",
        seed=seed,
    )

    # Back-transform to rain rates
    R_pred = transformation.dB_transform(R_normed_f, threshold=-10.0, inverse=True)[0]
    return R_pred  # (n_ens_members,num_timesteps,m,n)


def main(
        pre_settings,

        num_input_frames,
        ensemble_num,
        lead_time_steps,
        radolan_variable_name,
        mins_per_time_step,
        **__,
    ):

    R, radolan_dataset = load_data(**pre_settings)

    y = radolan_dataset[radolan_variable_name].y
    x = radolan_dataset[radolan_variable_name].x
    time = radolan_dataset[radolan_variable_name].time
    time = time[:-num_input_frames]

    pred_shape = (ensemble_num, lead_time_steps, len(time), len(y), len(x))
    R_pred = np.zeros(pred_shape)

    for i in range(len(R - num_input_frames)):
        idx_slice = (i, i + num_input_frames)
        R_pred_one_iteration = predict(R, idx_slice, **pre_settings)
        R_pred[:, :, i, :, :] = R_pred_one_iteration

    # Create a new dataset from the predictions
    steps_dataset = radolan_dataset.copy()
    steps_dataset.assign(steps_prediction=(('ensemble_num', 'lead_time_steps', 'time', 'y', 'x'), R_pred))

    steps_dataset = steps_dataset.assign_coords(
        ensemble_num=np.arange(R_pred.shape[0]),
        lead_time = np.arange(R_pred.shape[1]) * np.timedelta64(mins_per_time_step, 'm')

    )

    return steps_dataset


if __name__ == '__main__':
    pre_settings = {
        'ensemble_num': 20,
        'lead_time_steps': 6,
        'seed': 24,
        'mins_per_time_step': 5,
        'radolan_path': '/Users/jan/Programming/first_CNN_on_Radolan/dwd_nc/own_test_data/'
                           'testdata_two_days_2019_01_01-02.zarr',
        'radolan_variable_name': 'RV_recalc',
        'num_input_frames': 4

    }
    steps_dataset = main(pre_settings, **pre_settings)
    pass






