import os
import xarray as xr
import numpy as np
import dask.array as da
import dask

from pysteps import nowcasts
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.utils import transformation
from pysteps import motion

import sys
import io
import time
import concurrent.futures
from memory_profiler import profile
import multiprocessing as mp

import psutil
import threading
import matplotlib.pyplot as plt

sleep_time = 0.05
current_task = "red"
memory_usage = []

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
        start_time=None,
        end_time=None,
        **__
):
    # dataset = xr.open_dataset(radolan_path, engine='zarr') # , chunks=None # , chunks=None according to Sebastian more efficient as it avoids dask (default is chunks=1)
    dataset = xr.open_zarr(radolan_path, chunks='auto')
    # dataset = dataset.chunk({'step':1,'time':5,'y': 1200, 'x': 1100})
    
    # Uncomment this if only certain data range should be forecasted
    print(start_time, end_time)
    if start_time is not None and end_time is not None:
        dataset = dataset.sel(time=slice(start_time, end_time))
    # dataset = dataset.sel(time=slice('2019-01-01T12:00:00', '2019-01-01T18:30:00'))
    dataset = dataset.squeeze()

    # Set all negative values of the dataset to 0
    data_min = dataset.min(skipna=True, dim=None).RV_recalc.values
    if data_min < -0.1:
        raise ValueError(f'The min value of the dataset is {data_min}, which is below the arbitrary threshold of -0.1')

    dataset = dataset.where((dataset >= 0) | np.isnan(dataset), 0)
    # The where function keeps values where the condition is True and replaces the rest (where it's False)
    # with the value specified, in this case 0.
    print(f'min val in dataset is {dataset.min(skipna=True, dim=None).RV_recalc.values}')
    return dataset

def predict(
        radolan_slice: np.array,

        prediction_method,
        ensemble_num,
        lead_time_steps,
        seed,
        mins_per_time_step,
        radolan_path,
        num_input_frames,
        **__
):
    # global current_task
    # --- Processing ---
    # DB transform
    # current_task = "blue"
    radolan_slice_normed = transformation.dB_transform(radolan_slice, threshold=0.1, zerovalue=-15.0)[0]
    #print(f"__Shape of radolan_slice_normed: {radolan_slice_normed.shape}")
    # This returns a tuple with metatdata, where first entry is actual data

    # Estimate the motion field
    # current_task = "green"
    oflow_method = motion.get_method("LK")
    V = oflow_method(radolan_slice_normed)
    
    
    # current_task = "black"
    with SuppressPrint():
        nowcast_method = nowcasts.get_method(prediction_method) if prediction_method != "optical_flow" else None
        if prediction_method == "steps":
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
        elif prediction_method == "extrapolation":
            R_normed_f = nowcast_method(
                radolan_slice_normed[-1],    
                V,
                lead_time_steps,
            )
        elif prediction_method == "sprog":
            R_normed_f = nowcast_method(
                radolan_slice_normed,
                V,
                lead_time_steps,
                n_cascade_levels=6,
                precip_thr=-10.0,
            )
        elif prediction_method == "anvil":
            R_normed_f = nowcast_method(
                radolan_slice_normed,
                V,
                lead_time_steps,
                n_cascade_levels=6,
            )
        elif prediction_method == "optical_flow":
            return V.astype(np.float32)
        else:
            raise ValueError(f"Unknown prediction_method: {prediction_method}")
    
    # current_task = "red"
    # Back-transform to rain rates
    radolan_pred = transformation.dB_transform(R_normed_f, threshold=-10.0, inverse=True)[0]
    return radolan_pred  # (n_ens_members,num_timesteps,m,n)

def save_results_to_zarr(
    futures,
    save_zarr_path,
    dummy_dataset,
    ensemble_nums,
    lead_times,
    lead_time_steps,
    len_time,
    index,
    t0_of_radolan,
    attributes,
):
    for future_index, future in enumerate(futures):
        #print(f"Processing forecast for frame {index} out of {len_time}", flush=True)
        radolan_pred_one_iteration = future.result().astype(np.float32)
        time_value = dummy_dataset.time.values[index + future_index]

        if pre_settings['prediction_method'] == 'steps':
            radolan_pred_one_iteration = np.expand_dims(radolan_pred_one_iteration, axis=2)
            print(f"Shape of radolan_pred_one_iteration: {radolan_pred_one_iteration.shape}")
            radolan_pred_ds = xr.Dataset(
                data_vars={
                    'steps': (('ensemble_num', 'lead_time', 'time', 'y', 'x'), radolan_pred_one_iteration)
                },
                coords={
                    'ensemble_num': ensemble_nums,
                    'lead_time': lead_times,
                    'time': [time_value],
                    'y': dummy_dataset.y,
                    'x': dummy_dataset.x,
                }
            )
        elif pre_settings['prediction_method'] == 'optical_flow':
            radolan_pred_one_iteration = np.expand_dims(radolan_pred_one_iteration, axis=1)
            radolan_pred_ds = xr.Dataset(
                data_vars={
                    pre_settings['prediction_method']: (('uv', 'time', 'y', 'x'), radolan_pred_one_iteration)
                },
                coords={
                    'uv': ['u', 'v'],
                    'time': [time_value],
                    'y': dummy_dataset.y,
                    'x': dummy_dataset.x,
                }
            )
        else:
            radolan_pred_one_iteration = np.expand_dims(radolan_pred_one_iteration, axis=1)
            radolan_pred_ds = xr.Dataset(
                data_vars={
                    pre_settings['prediction_method']: (('lead_time', 'time', 'y', 'x'), radolan_pred_one_iteration)
                },
                coords={
                    'lead_time': lead_times,
                    'time': [time_value],
                    'y': dummy_dataset.y,
                    'x': dummy_dataset.x,
                }
            )

        radolan_pred_ds['time'].encoding['units'] = f'minutes since {t0_of_radolan}'
        radolan_pred_ds['time'].encoding['dtype'] = 'float64'
        radolan_pred_ds.attrs = attributes
        
        if pre_settings['prediction_method'] == 'optical_flow':
            radolan_pred_ds = radolan_pred_ds.chunk({'uv': 2, 'time': 50, 'y': len(dummy_dataset.y), 'x': len(dummy_dataset.x)})
        elif pre_settings['prediction_method'] == 'steps':
            radolan_pred_ds = radolan_pred_ds.chunk({'ensemble_num': min(6, len(ensemble_nums)), 'lead_time': min(2, len(lead_time_steps)), 'time': 50, 'y': len(dummy_dataset.y), 'x': len(dummy_dataset.x)})
        else:
            radolan_pred_ds = radolan_pred_ds.chunk({'lead_time': min(2, len(lead_time_steps)), 'time': 50, 'y': len(dummy_dataset.y), 'x': len(dummy_dataset.x)})
        
        if index + future_index == 0:
            print(f"Creating zarr file: {save_zarr_path}", flush=True)
            radolan_pred_ds.to_zarr(save_zarr_path, mode='w')
        else:
            # print(f"Appending to zarr file: {save_zarr_path}", flush=True)
            radolan_pred_ds.to_zarr(save_zarr_path, mode='a-', append_dim='time')

# @profile
def main(
    pre_settings,
    num_input_frames,
    ensemble_num,
    lead_time_steps,
    radolan_variable_name,
    mins_per_time_step,
    save_zarr_path,
    start_date=None,
    end_date=None,
    **__,
):
    global current_task
    lead_time_steps = np.arange(lead_time_steps)

    start_date = np.datetime64(start_date)
    end_date = np.datetime64(end_date)
    days = np.arange(start_date, end_date, np.timedelta64(1, 'D'))
    
    for current_day in days:
        
        day_start = current_day
        day_end = current_day + np.timedelta64(1, 'D') + np.timedelta64(15, 'm')
        if day_end > end_date:
            day_end = end_date
            
        save_zarr_path = f"{pre_settings['prediction_method']}_{str(day_start)[:10]}.zarr"
        print(f'Processing day: {day_start} to {day_end}')
        
        radolan_dataset = load_data(**pre_settings, start_time=day_start, end_time=day_end)
    
        y = radolan_dataset[radolan_variable_name].y
        x = radolan_dataset[radolan_variable_name].x
        time_dim = radolan_dataset[radolan_variable_name].time

        len_time = len(time_dim) - num_input_frames

        t0_of_radolan = str(radolan_dataset.time.values[0])

        dummy_dataset = radolan_dataset.copy()
        dummy_dataset = dummy_dataset.isel(time=slice(0, -num_input_frames))

        ensemble_nums = np.arange(ensemble_num)
        lead_times = (lead_time_steps * np.timedelta64(mins_per_time_step, 'm')
                    .astype('timedelta64[ns]'))

        print('Creating forecast')
        workers = 4
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            batch_size = 50
            
            index = 0
            for batch_start in range(0, len_time, batch_size):
                batch_end = min(batch_start + batch_size, len_time)
                print(f"Batch from {batch_start} to {batch_end}")
                futures = [
                    executor.submit(predict, radolan_dataset.isel(
                        time=slice(i, i + num_input_frames)
                    )[radolan_variable_name].values, **pre_settings)
                    for i in range(batch_start, batch_end)
                ]

                futures_len = len(futures)
                save_results_to_zarr(
                    futures,
                    save_zarr_path,
                    dummy_dataset,
                    ensemble_nums,
                    lead_times,
                    lead_time_steps,
                    len_time,
                    index,
                    t0_of_radolan,
                    radolan_dataset.attrs
                )
                index += futures_len


if __name__ == '__main__':
    """
    Data format in the end:
    every forecast has the time stamp of the first input frame.
    So in order top get the actual time of the forcast calculate:
    time_stamp + input_frames + lead time (starting at 0)
    Each forcast has ensemble_num dimension
    and lead_time dimension 
    """
    # start tracking memory usage
    stop_event = threading.Event()
    def track_memory_usage():
        global current_task
        process = psutil.Process()
        while not stop_event.is_set():
            # Get memory info for this Python process and all its threads
            process_memory = process.memory_info().rss / 1024**2  # Convert bytes to MB
            
            children_memory = 0
            for child in process.children(recursive=True):
                children_memory += child.memory_info().rss / 1024**2
            total_memory = process_memory + children_memory
            # total_memory = psutil.virtual_memory().used / 1024**2
            memory_usage.append((current_task, total_memory))
            time.sleep(sleep_time)
    tracking_thread = threading.Thread(target=track_memory_usage, daemon=True)
    tracking_thread.start()
    
    start_time = time.time()
    start_date = '2019-01-01T12:00:00'
    end_date = '2019-01-03T12:30:00'
    pre_settings = {

        'prediction_method': "steps",  # steps, extrapolation, sprog, anvil or optical_flow
        'ensemble_num': 2, #20, #20,   # only used for steps
        'lead_time_steps': 3, # 12, #6,
        'seed': 24,
        'mins_per_time_step': 5,
        'radolan_variable_name': 'RV_recalc',
        'num_input_frames': 4,
        # -- local testing ---
        'radolan_path': './testdata_two_days_2019_01_01-02.zarr',
        'save_zarr_path': './testdata_two_days_2019_01_01-02_sprog.zarr',
        
        # -- big dataset cluster --
        # 'radolan_path': '/mnt/qb/work2/butz1/bst981/weather_data/dwd_nc/zarr/RV_recalc.zarr',
        # 'save_zarr_path': '/mnt/qb/work2/butz1/bst981/weather_data/steps_forecasts/steps_forecasts_rv_recalc_5_ens_members.zarr',
        # -- test dataset cluster --
        # 'radolan_path': '/mnt/qb/work2/butz1/bst981/first_CNN_on_Radolan/dwd_nc/own_test_data/testdata_two_days_2019_01_01-02.zarr',
        # 'save_zarr_path': '/mnt/qb/work2/butz1/bst981/weather_data/steps_forecasts/steps_forecast_testdata_two_days_2019_01_01-02_token.zarr',
    }
    main(pre_settings, **pre_settings, start_date=start_date, end_date=end_date)
    
    total_time = time.time() - start_time
    print(f'\nTotal time: {total_time}')
    
    # # Execute to check created file:
    # try: 
    #     dataset_predictions = xr.open_dataset("extrapolation_2019-01-01.zarr", engine='zarr')
    #     print('\nSaved Zarr file loaded successfully')
    #     print(dataset_predictions)
    # except: 
    #     print('Error loading zarr file')
    
    # Memory tracking evaluation
    stop_event.set()
    tracking_thread.join()
    # print out the max memory usage
    memory_values = [x[1] for x in memory_usage]  # Extract memory values
    marker_colors = [x[0] for x in memory_usage]
    time_values = [i * sleep_time for i in range(len(memory_values))]
    print("Max memory usage: ", max(memory_values), " MB. Median memory usage: ", np.median(memory_values), " MB.")
    
    plt.figure(figsize=(12, 8))
    # scatter = plt.scatter(range(len(memory_values)), memory_values, c=marker_colors, label='Memory Usage (MB)', marker='o')
    for i in range(1, len(memory_values)):
       plt.plot([time_values[i-1], time_values[i]], 
                [memory_values[i-1], memory_values[i]], 
                 color=marker_colors[i], linewidth=2)

    plt.title('Memory Usage Over Time')  # Add a title
    plt.xlabel('Time (seconds)')  # Label x-axis
    plt.ylabel('Memory Usage (MB)')  # Label y-axis
    plt.grid(True)
    # round up the max memory value to the next thousand
    plt.ylim(0, np.ceil(max(memory_values) / 1000) * 1000)
    plt.savefig('memory_usage.png')
    
    






