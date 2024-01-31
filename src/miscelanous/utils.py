import os
import pandas as pd

def make_directory(dir_path):
    ''' Create a directory at the specified path if it doesn't already exist '''
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory created: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")
    except Exception as e:
        print(f"An error occurred while creating the directory: {e}")


def save_data_csv(filepath: str, data: list[list[float]], append_mode: bool = True):
    mode = 'a' if append_mode else 'w'
    header = False if append_mode else True
    df = pd.DataFrame(data)
    df.to_csv(filepath, mode=mode, header=header, index=False)


def load_data_csv(filepath: str) -> list[list[float]]:
    df = pd.read_csv(filepath, header=0, dtype='float32')
    return df.values.tolist()


def remove_file_if_exists(filepath: str):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"File '{filepath}' has been removed.")
        else:
            print(f"No file found at '{filepath}'. Nothing to remove.")
    except Exception as e:
        print(f"An error occurred while removing the file: {e}")

