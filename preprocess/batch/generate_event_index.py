import glob
import os
import pandas as pd
from tqdm import tqdm
import pickle


if __name__ == "__main__":
    directory_path = "../../data/preprocessed"
    output_dir = "../../data/preprocessed"

    files = sorted(glob.glob(os.path.join(directory_path, "events-*.pkl")))

    index = {}
    for filename in tqdm(files, ncols=100, desc="Processing"):
        file_path = os.path.join(directory_path, filename)
        file_name = os.path.splitext(filename)[0]

        df = pd.read_pickle(file_path)
        index[file_name] = set(df.index)

    # save index to preprocessed folder as a pickle file
    with open(os.path.join(output_dir, "event_index.pkl"), "wb") as f:
        pickle.dump(index, f)
