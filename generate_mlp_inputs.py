"""
Assumes water level data is 6-minute and wind data is hourly.
"""

# Imports.
import os
import datetime
import pandas as pd
import numpy as np

# From repo.
import helpers


# Get paths to preprocessed data from config_dataset_generation.
config = helpers.load_configs("config_mlp_input_generation.json")
file_paths = config['file_paths']

# Get data files.
# Store datetime, water level, wind from paired station, and water level from primary station in data frame.
# Parse datetimes and set as index.
input_wl_data = pd.read_csv(file_paths['paired_station_wl_data'], parse_dates=True, index_col=0)
# input_wind_data = pd.read_csv(file_paths['paired_station_wind_data'], parse_dates=True, index_col=0)
target_wl_data = pd.read_csv(file_paths['target_station_wl_data'], parse_dates=True, index_col=0)

# all_data_temp
all_data = pd.merge(target_wl_data, input_wl_data, left_index=True, right_index=True, how='left',
                         suffixes=('_target', '_input'))
# all_data = pd.merge(all_data_temp, input_wind_data, left_index=True, right_index=True, how='left')

# Store column names for slicing later.
# input_wind_cols = input_wind_data.columns.tolist()
input_wl_col = input_wl_data.columns[0]
target_col = target_wl_data.columns[0]

# Forward fill 6-minute slots with hourly wind. Limit forward filling up to 9 points to only represent the hour.
# all_data['u'], all_data['v'] = all_data['u'].ffill(limit=9), all_data['v'].ffill(limit=9)

# All rows where all columns are non-NaN.
# valid_df = all_data[all_data[all_data.columns].notna().all(axis=1)].copy()
valid_df = all_data.dropna().copy()

# Group consecutive 6 minute segments into blocks.
valid_df['time_diff'] = valid_df.index.to_series().diff().fillna(pd.Timedelta(minutes=6))
is_regular = valid_df['time_diff'] == pd.Timedelta(minutes=6)

# Group ID changes when time jumps are irregular.
group_id = (is_regular != is_regular.shift()).cumsum()
valid_df['group'] = group_id

# Size of groups will be compared to window sizes of input data per sample.
group_sizes = valid_df.groupby('group').size()

# Get window size from config. This is how many past/future data points are being used to predict the current
# point.
window_size_past_list = config['output_settings']['window_size_past']
window_size_future_list = config['output_settings']['window_size_future']

# Store wl and wind window sizes. Convert window size for wind in hours to 6-minute.
wl_past_window_size = window_size_past_list[0]
wind_past_window_size = window_size_past_list[1] * 10
wl_future_window_size = window_size_past_list[0]
wind_future_window_size = window_size_past_list[1] * 10

max_past_window_size = max(wl_past_window_size, wind_past_window_size)
max_future_window_size = max(wl_future_window_size, wind_future_window_size)

# Smallest data segment size required per sample. +1 for the target value.
min_required_group_size = max_past_window_size + max_future_window_size + 1

# Get groups long enough to map inputs to a target sample.
valid_groups = group_sizes[group_sizes >= min_required_group_size]

# Iterate over all segments, getting inputs (past and future 6-min water levels + past hourly wind) in slices to 
# store in array.
inputs_arr = []
targets_arr = []
times_arr = []

for group_id in valid_groups.index:
    # Stop adding samples when desired length of dataset is achieved.
    if len(targets_arr) >= config['output_settings']['dataset_length_days_request'] * 24 * 6:
        break

    group_df = valid_df[valid_df['group'] == group_id]
    # group_df = group_df.sort_index()  # Ensure chronological order.

    # Convert columns to numpy for faster slicing.
    wl_values = group_df[input_wl_col].to_numpy()
    # wind_values = group_df[input_wind_cols].to_numpy()
    targets = group_df[target_col].to_numpy()
    timestamps = group_df.index.to_numpy()

    # Get all samples in the segment that are bounded by full past/future windows.
    # Extract wind values one hour (10 points) apart.
    for i in range(max_past_window_size, len(group_df) - max_future_window_size):
        wl_past = wl_values[i - wl_past_window_size: i]
        wl_future = wl_values[i + 1: i + 1 + wl_future_window_size]
        # wind_past = wind_values[i - wind_past_window_size: i: 10]
        # wind_future = wind_values[i + 1: i + 1 + wind_future_window_size: 10]

        # Flatten wind (2D) and concatenate everything into a 1D input.
        input_seq = np.concatenate([wl_past, wl_future])#, wind_past.flatten()])

        target_val = targets[i]
        target_time = timestamps[i]

        inputs_arr.append(input_seq)
        targets_arr.append(target_val)
        times_arr.append(target_time)
    # End for.
# End for.

# Save mlp inputs (including inputs and target) to csv. Target should be last column.
X = np.array(inputs_arr)
y = np.array(targets_arr)
times = pd.to_datetime(times_arr)

output_df = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), index=times)
print(output_df.shape)
output_df.to_csv(file_paths['output'], index=True)










