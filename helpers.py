import json


def load_configs(file_path):
    try:
        with open(file_path, 'r') as file:
            user_config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Config file '{file_path}' not found.")
    return user_config














