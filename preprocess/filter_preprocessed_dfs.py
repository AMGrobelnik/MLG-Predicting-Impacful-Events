from util.iterate_files import iterate_files
import pickle
import os
import pandas as pd
from IPython.display import display
from tabulate import tabulate
from tqdm import tqdm

# Load file B_recent_10_khops_3k.pkl from batch_ids
file_path = "./batch/batch_ids/B_recent_10_khops_3k.pkl"
with open(file_path, "rb") as file:
    batches = pickle.load(file)

# Merge all string elements appearing in all batches into 1 set
batch_ids_set = set()
for target_ids, similar_ids in batches:
    batch_ids_set.update(target_ids)
    batch_ids_set.update(similar_ids)


# Path to the directory containing the .pkl files
directory = "../data/preprocessed/"

# Iterate over all files in the directory
for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".pkl"):
        file_path = os.path.join(directory, filename)

        # Load the .pkl file
        with open(file_path, "rb") as file:
            preprocessed_df = pickle.load(file)

        # Filter the dataframe to only include events in batch_ids_set
        preprocessed_df = preprocessed_df[preprocessed_df.index.isin(batch_ids_set)]

        # Iterate over similarEvents and remove those not in batch_ids_set
        preprocessed_df["similarEvents"] = preprocessed_df["similarEvents"].apply(
            lambda x: [event for event in x if event["uri"] in batch_ids_set]
        )

        # Create a new folder called preprocessed_filtered
        filtered_directory = "../data/preprocessed_filtered/"
        os.makedirs(filtered_directory, exist_ok=True)

        # Save the preprocessed dataframe in the filtered directory with the same filename
        filtered_filename = "filtered_" + filename
        filtered_file_path = os.path.join(filtered_directory, filtered_filename)
        with open(filtered_file_path, "wb") as file:
            pickle.dump(preprocessed_df, file)