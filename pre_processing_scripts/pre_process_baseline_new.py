import os
import xarray as xr
import numpy as np
import dask.array as da

from pysteps import nowcasts
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.utils import transformation

import sys
import io
import time


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

    # Uncomment this if only certain data range should be forecasted
    dataset = dataset.sel(time=slice('2019-01-01T12:00:00', '2019-01-01T12:30:00'))

    return dataset


def predict(
        radolan_slice: np.array,

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
    radolan_slice_normed = transformation.dB_transform(radolan_slice, threshold=0.1, zerovalue=-15.0)[0]
    # This returns a tuple with metatdata, where first entry is actual data

    # Estimate the motion field
    V = dense_lucaskanade(radolan_slice_normed)
    # The STEPS nowcast
    nowcast_method = nowcasts.get_method("steps")
    R_normed_f = nowcast_method(
        radolan_slice_normed,
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
    radolan_pred = transformation.dB_transform(R_normed_f, threshold=-10.0, inverse=True)[0]
    return radolan_pred  # (n_ens_members,num_timesteps,m,n)


def main(
        pre_settings,

        num_input_frames,
        ensemble_num,
        lead_time_steps,
        radolan_variable_name,
        mins_per_time_step,
        **__,
    ):

    radolan_dataset = load_data(**pre_settings)

    y = radolan_dataset[radolan_variable_name].y
    x = radolan_dataset[radolan_variable_name].x
    time_dim = radolan_dataset[radolan_variable_name].time

    # The length of the time dimension is len(R) - num_input_frames + 1, as R[0 : num_input_frames] already produces
    # a forecast: R[num_input_frames] = forecast(R[0 : num_input_frames])
    # Say num_input_frames = 4, then we would have to remove only 3 time steps to fit the forecast data into the time
    # dimension
    len_time = len(time_dim) - num_input_frames + 1

    pred_shape = (ensemble_num, lead_time_steps, len_time, len(y), len(x))

    dummy_dataset = radolan_dataset.copy()

    # We are taking the original data set and start at the 'num_input_frames'th frame time-wise
    # as the first num_input_frames are used to make the prediction
    dummy_dataset = dummy_dataset.isel(time=slice(num_input_frames-1, None))

    ensemble_nums = np.arange(ensemble_num)
    lead_time_steps = np.arange(lead_time_steps)+1  # We start counting at 1, as first lead time is i.e. 5 mins and not 0
    lead_times = (lead_time_steps * np.timedelta64(mins_per_time_step, 'm')
                  .astype('timedelta64[ns]'))


    steps_dataset = xr.Dataset(
        # dims=('ensemble_num', 'lead_time', 'time', 'y', 'x'),
        coords={'ensemble_num': ensemble_nums,
                'lead_time': lead_times,
                'time': dummy_dataset.time.values,
                'y': dummy_dataset.y.values,
                'x': dummy_dataset.x.values}
    )
    steps_dataset.to_zarr(pre_settings['save_zarr_path'], mode='w')


    print('Creating STEPS forecast')
    for i in range(len_time):
        print(f'Forecast for frame {i} to {i} + 4')
        with SuppressPrint():

            radolan_slice = radolan_dataset.isel(
                time=slice(i, i + num_input_frames)
            )
            radolan_vals = radolan_slice[radolan_variable_name].values
            radolan_pred_one_iteration = predict(radolan_vals, **pre_settings)
            radolan_pred_one_iteration = np.expand_dims(radolan_pred_one_iteration, axis=2)

            time_value = dummy_dataset.time.isel(time=i).values

            radolan_pred_da = xr.DataArray(
                radolan_pred_one_iteration,
                dims=('ensemble_num', 'lead_time', 'time', 'y', 'x'),
                # coords={
                #    'ensemble': ensemble_nums,
                #    'lead_time': lead_times,
                #    'time': [time_value],
                #    'y': dummy_dataset.y,
                #    'x': dummy_dataset.x,
                # },
                name='steps'
            )
            radolan_pred_da.to_zarr(pre_settings['save_zarr_path'], mode='a', append_dim='time')




if __name__ == '__main__':
    pre_settings = {
        'ensemble_num': 1, #20, #20,
        'lead_time_steps': 1, # 24, #6,
        'seed': 24,
        'mins_per_time_step': 5,
        'radolan_variable_name': 'RV_recalc',
        'num_input_frames': 4,
        # -- local testing ---
        'radolan_path': '/Users/jan/Programming/first_CNN_on_Radolan/dwd_nc/own_test_data/'
                           'testdata_two_days_2019_01_01-02.zarr',
        'save_zarr_path': '/Users/jan/Downloads/'
                           'testdata_two_days_2019_01_01-02_steps_predictions.zarr',
        # -- big dataset cluster --
        # 'radolan_path': '/mnt/qb/work2/butz1/bst981/weather_data/dwd_nc/zarr/RV_recalc.zarr',
        # 'save_zarr_path': '/mnt/qb/work2/butz1/bst981/weather_data/steps_forecasts/steps_forecasts_rv_recalc.zarr',
        # -- test dataset cluster --
        # 'radolan_path': '/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/dwd_nc/own_test_data/testdata_two_days_2019_01_01-02.zarr',
        # 'save_zarr_path': '/mnt/qb/work2/butz1/bst981/weather_data/dwd_nc/zarr/steps_forecast_testdata_two_days_2019_01_01-02.zarr',
    }
    start_time = time.time()
    main(pre_settings, **pre_settings)
    total_time = time.time() - start_time
    print(f'Time of script: {total_time}')







