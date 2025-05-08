"""
This config file is for pre_process_baseline.py and pre_process_baseline_daily.py
"""

start_date = "2020-06-01T00:00:00"
end_date = "2020-07-01T00:15:00"
workers = 6 # not more than 8
batch_size = workers * 4 # 50

settings = {
    # Input
    "radolan_path": "../precipitation_radolan/RV_recalc.zarr",
    # Output
    # Will only be used for pre_process_baseline.py
    "save_zarr_path": "./steps_2020-06.zarr",
    #
    "radolan_variable_name": "RV_recalc",
    #
    "prediction_method": "steps", # steps, extrapolation, sprog, anvil or optical_flow
    "ensemble_num": 24, #20, #20,   # only used for steps
    "lead_time_steps": 3, # 12, #6,
    "seed": 24,
    "mins_per_time_step": 5,
    "num_input_frames": 4,
}
