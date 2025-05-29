import json
import os
import pandas as pd
import numpy as np


def load_configs(file_path):
    try:
        with open(file_path, 'r') as file:
            user_config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Config file '{file_path}' not found.")
    return user_config


def verify_path(path, flag=[True]):
    if not os.path.exists(path):
        flag[0] = False

    return path


def read_file_to_df(file, index_limit=None, flag=[False], error=[""]):

    # Initialize df to empty dataframe.
    df = pd.DataFrame()

    # Find where valid data stops using end_file_index().
    if index_limit is None:
        index_limit = end_file_index(file)

    # Read the file into a dataframe. If error reading or finding file, record the error.
    if os.path.isfile(file):
        try:
            df = pd.read_csv(file, nrows=index_limit)
            flag[0] = True
        except Exception as e:
            error[0] = str(e)
            flag[0] = False
    else:
        error[0] = "File not found."
        flag[0] = False
    # End for.

    return df
# End read_file_to_df.


def end_file_index(filename):

    line_count = 0
    trailing_lines = 0

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line_count += 1
            if line.startswith("# "):
                trailing_lines += 1
    # Close file.

    valid_data_lines = line_count - trailing_lines - 1  # -1 for last new line.

    return valid_data_lines
# End end_file_index().


def clean_dataframe(df, drop=[]):
    # Replace missing values with NaN.
    df.replace([-999, -99, 99, 'NA', 'RM'], np.nan, inplace=True)

    # Assume first column should be converted to datetimes.
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')

    # Convert other cols to numeric and drop undesired columns.
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if drop:
        df = df.drop(columns=[df.columns[i] for i in drop])

    return df
# End clean_dataframe.










