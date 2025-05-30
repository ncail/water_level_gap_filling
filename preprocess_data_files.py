"""
Script for preprocessing data downloaded from Lighthouse meant to create input for the gap-filling neural net.
Program replaces RM, NA, etc. with NaNs and converts timestamps and values to pandas standards.
Drops specified columns using position indexing from config file.
"""

# Import from repo.
import helpers

# Python library imports.
import os


# Program start.

# Get configs.
config = helpers.load_configs('config_preprocess_data_files.json')

# Check that paths are valid. Read into dataframes.
data_paths = config['data_paths']
args_flag_ptr = [True]
dataframes = []
for path in data_paths:
    path = helpers.verify_path(path, flag=args_flag_ptr)

    if args_flag_ptr[0] is False:
        raise FileNotFoundError(f"Path <{path}> does not exist. Exiting program.")

    df = helpers.read_file_to_df(path)
    dataframes.append(df)
# End for.

# Get output path.
output_path = config['output_dir']
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Clean dataframes: parse dates, convert to numeric, convert errors to NaN, drop undesired columns. Send to csv.
drop_cols = config['drop_columns']
output_filenames = config['output_filenames']
for idx, df in enumerate(dataframes):
    df = helpers.clean_dataframe(df, drop=drop_cols)
    df.to_csv(f'{output_path}/{output_filenames[idx]}', index=False)
# End for.

# End program.
