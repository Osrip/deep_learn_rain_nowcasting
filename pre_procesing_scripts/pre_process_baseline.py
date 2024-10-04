import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pysteps import nowcasts, rcparams
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.utils import conversion, dimension, transformation
from pysteps.visualization import plot_precip_field
from tqdm import tqdm
import sys
import io


# Temporarily redirect stdout to suppress print statements
class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stdout after exiting the block
        sys.stdout = self._original_stdout

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
    dataset = dataset.sel(time=slice('2019-01-01T12:00:00', '2019-01-01T12:30:00'))

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
        precip_thr=-10.0,
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
    time = time[num_input_frames:]

    pred_shape = (ensemble_num, lead_time_steps, len(time), len(y), len(x))
    R_pred = np.zeros(pred_shape)

    with SuppressPrint():
        for i in tqdm(range(len(R) - num_input_frames)):
            idx_slice = (i, i + num_input_frames)
            R_pred_one_iteration = predict(R, idx_slice, **pre_settings)
            R_pred[:, :, i, :, :] = R_pred_one_iteration


    # Create a new dataset from the predictions
    steps_dataset = radolan_dataset.copy()

    # We are taking the original data set and start at the 'num_input_frames'th frame time-wise
    # as the first num_input_frames are used to make the prediction
    steps_dataset = steps_dataset.isel(time=slice(num_input_frames, None))

    # Add the predictions with appropriate dimensions
    # First, ensure ensemble_num and lead_time coordinates are added to the dataset
    ensemble_nums = np.arange(R_pred.shape[0])
    lead_times = ((np.arange(R_pred.shape[1])+1) * np.timedelta64(mins_per_time_step, 'm')
                  .astype('timedelta64[ns]'))

    # Assign these coordinates to the dataset
    steps_dataset = steps_dataset.assign_coords(ensemble_num=ensemble_nums, lead_time=lead_times)

    # We need to make sure that we expand `time`, `y`, and `x` to match the dimensions of `R_pred`
    # This assumes `R_pred` shape is [ensemble_num, lead_time, time, y, x]
    steps_prediction = xr.DataArray(
        R_pred, dims=('ensemble_num', 'lead_time', 'time', 'y', 'x'),
        coords={'ensemble_num': ensemble_nums,
                'lead_time': lead_times,
                'time': steps_dataset.time,
                'y': steps_dataset.y,
                'x': steps_dataset.x}
    )

    # Assign the data array to the dataset
    steps_dataset = steps_dataset.assign(steps_prediction=steps_prediction)
    # Im Moment werden frames aus min 0, 5, 10, 15 genommen, prediction gemacht und dann die Prediction
    # (mit mehreren lead times) auf min 20 assigned.
    # Das heißt das ground truth von min 20 genau die prediction von min 20 mit lead time 0=5 ist.
    # Was passiert mit der letzten prediction? Eigentlich müstte für die ja ein neuer time stamp erfunden werden, die ground truth nicht hat.
    # Eigentlich wäre es intuitiver die prediction der minute 15 zu assignen, also dem letzten Bild, das genutzt wurde um die
    # prediction zu machen.
    return steps_dataset


if __name__ == '__main__':
    pre_settings = {
        'ensemble_num': 3, #20,
        'lead_time_steps': 2, #6,
        'seed': 24,
        'mins_per_time_step': 5,
        'radolan_path': '/Users/jan/Programming/first_CNN_on_Radolan/dwd_nc/own_test_data/'
                           'testdata_two_days_2019_01_01-02.zarr',
        'radolan_variable_name': 'RV_recalc',
        'num_input_frames': 4

    }

    steps_dataset = main(pre_settings, **pre_settings)
    pass






