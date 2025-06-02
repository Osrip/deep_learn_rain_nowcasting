
# Select last entry of groups
for key, group_ds in binned_patches:
    group_ds = group_ds.RV_recalc

# select one entry
bla = group_ds.isel(
stacked_time_y_outer_y_inner_x_outer_x_inner = 0
)

# the dataarrays have all index info in stacked_time_y_outer_y_inner_x_outer_x_inner

bla.stacked_time_y_outer_y_inner_x_outer_x_inner

"""
Out[16]: 
<xarray.DataArray 'stacked_time_y_outer_y_inner_x_outer_x_inner' ()> Size: 8B
array((Timestamp('2019-01-01 08:35:00'), 9, 22, 10, 9), dtype=object)
Coordinates:
    latitude                                      float64 8B 53.45
    longitude                                     float64 8B 6.868
    missing_data_RV_recalc                        float32 4B 0.0
    step                                          timedelta64[ns] 8B 00:00:00
    x                                             float64 8B -214.5
    y                                             float64 8B -3.92e+03
    stacked_time_y_outer_y_inner_x_outer_x_inner  object 8B (Timestamp('2019-...
    time                                          datetime64[ns] 8B 2019-01-0...
    y_outer                                       int64 8B 9
    y_inner                                       int64 8B 22
    x_outer                                       int64 8B 10
    x_inner                                       int64 8B 9
"""

