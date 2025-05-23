# Script for preprocessing data files.
# Cleans files, and converts them to Pandas standards.
# Outputs preprocessed files to 'preprocessed' directory inside original data paths.

import helpers

import os
import sys
import glob
import pandas as pd


def main(args):
    # Get configs.
    config = helpers.load_configs(args.config)

    # Get paths to files.
    config_refdir = config['data']['paths']['refdir']
    config_primarydir = config['data']['paths']['primarydir']
    args_flag_ptr = [True]
    paths = helpers.get_data_paths(args, config_refdir, config_primarydir, flag=args_flag_ptr)

    # Get output path.
    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Check that get_data_paths succeeded.
    if args_flag_ptr[0] is False:
        print("args_flag_ptr is False. Path(s) do not exist. Exiting program.")
        sys.exit()

    # Get positions of datetime and water level columns in order of: ref, primary.
    dt_col_pos = [config['data']['columns']['ref_dt_pos'], config['data']['columns']['primary_dt_pos']]
    pwl_col_pos = [config['data']['columns']['ref_wl_pos'], config['data']['columns']['primary_wl_pos']]

    # Get header information in order of ref, primary.
    isHeader = [config['data']['headers']['ref'], config['data']['headers']['primary']]

    # Loop over paths if they were entered. paths contains refdir, then primarydir, so uses the column positions in
    # this order.
    for loop in range(2):
        if paths[loop]:
            # Get all csv files from primary path.
            csv_files = glob.glob(f"{paths[loop]}/*.csv")

            # Get header info.
            if isHeader[loop]:
                df_first = pd.read_csv(csv_files[0])
                column_names = df_first.columns

            # Read files to a concatd dataframe, then split by year.
            df_list = []
            for file in csv_files:
                try:
                    df_from_file = pd.read_csv(file, header=None)
                except pd.errors.EmptyDataError:
                    print(f"<{file}> is empty. Skipping to next file.")
                    continue
                if isHeader[loop]:
                    df_from_file.columns = column_names

                df_list.append(df_from_file)
            # End for.

            df_concat = pd.concat(df_list, ignore_index=True)
            split_df_dict = helpers.split_by_year(df_concat, df_concat.columns[dt_col_pos[loop]])

            # Clean and output dataframes to csv.
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for year, df in split_df_dict.items():
                cleaned_df = helpers.clean_dataframe(df, df.columns[dt_col_pos[loop]], df.columns[pwl_col_pos[loop]])

                if not pd.isna(year):
                    cleaned_df.to_csv(f'{output_path}/{year}.csv', index=False)
            # End for.
        # End if.
    # End for.
# End main.


if __name__ == "__main__":
    main_args = helpers.parse_arguments()
    main(main_args)
