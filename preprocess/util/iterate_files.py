import os
from glob import glob

import pandas as pd
from tqdm import tqdm


def iterate_files(action, directory_path: str, n=3000):
    """
    Iterates over files in a directory and applies an action to each file
    :param action: (file_name, df) -> None
    :param directory_path: file directory (preprocessed dataframes)
    :param n: number of files to process
    :return:
    """
    files = sorted(glob(os.path.join(directory_path, "events-*.pkl")))
    files = files[:n]

    for filename in tqdm(files, ncols=100, desc="Processing"):
        file_path = os.path.join(directory_path, filename)
        file_name = os.path.splitext(filename)[0]

        df = pd.read_pickle(file_path)
        action(file_name, df)
