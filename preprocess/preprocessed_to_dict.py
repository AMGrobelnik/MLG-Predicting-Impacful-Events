import glob
import pandas as pd
from tqdm import tqdm
import pickle
import json

def files_to_dict(files):
    for file in tqdm(files, ncols=100, desc="Processing"):
        df = pd.read_pickle(file)
        info, similar_events = df["info"], df["similarEvents"]
        index = df.index

        df_dict = {}
        for i, e in enumerate(index):
            df_dict[e] = {
                "info": info[i],
                "similar_events": similar_events[i],
            }

        file_name = file.split("/")[-1]
        file_name = file_name.split('\\')[-1]
        with open(f"../data/preprocessed_dicts/{file_name}", "wb") as f:
            pickle.dump(df_dict, f)


def main():
    files = glob.glob("../data/preprocessed/events-*.pkl")

    files_to_dict(files)


if __name__ == "__main__":
    main()
