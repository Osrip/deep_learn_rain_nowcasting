import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def load_data_from_logs(s_calc_baseline, s_dirs,
    rel_path_train='logs/train_log/version_0/metrics.csv',
    rel_path_val='logs/val_log/version_0/metrics.csv',
    rel_path_base_train='logs/base_train_log/version_0/metrics.csv',
    rel_path_base_val='logs/base_val_log/version_0/metrics.csv',
    **__):

    save_dir = s_dirs['save_dir']

    train_df = pd.read_csv('{}/{}'.format(save_dir, rel_path_train))
    val_df = pd.read_csv('{}/{}'.format(save_dir, rel_path_val))
    if s_calc_baseline:
        base_train_df = pd.read_csv('{}/{}'.format(save_dir, rel_path_base_train))
        base_val_df = pd.read_csv('{}/{}'.format(save_dir, rel_path_base_val))
    else:
        base_train_df = None
        base_val_df = None

    return train_df, val_df, base_train_df, base_val_df


def interpolate_smooth(x, y, window_size_smooth=4, polynomial_order_smooth=3, num_data_points_interp=4000, smooth=True, interpolate=True):
    '''
    Interpolating and smoothing for line plots
    For smoothing:
    polyorder must be less than window_length
    window_length must be less than or equal to the size of x
    Assumes linspace of x and accordingly ordered y

    '''
    if smooth:
        y = savgol_filter(y, window_size_smooth, polynomial_order_smooth)  # window size, polynomial order
    if interpolate:
        f_interpolate = interp1d(x, y, kind='cubic')
        x = np.linspace(np.min(x), np.max(x), num=num_data_points_interp, endpoint=True)
        y = f_interpolate(x)
    return x, y

