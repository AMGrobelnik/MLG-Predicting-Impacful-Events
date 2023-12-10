import os
import shutil
import glob
import random



def clear_directory(directory):
    """Delete all .pkl files in the specified directory"""
    for file in glob.glob(os.path.join(directory, "*.pkl")):
        os.remove(file)

def create_splits_for_folder(folder_name):
    """Create train, val, test splits for a given folder"""
    folder_path = os.path.join(base_dir, folder_name)
    train_dir = os.path.join(folder_path, "train")
    val_dir = os.path.join(folder_path, "val")
    test_dir = os.path.join(folder_path, "test")

    # Make sure train, val, and test directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Clear existing files in the directories
    clear_directory(train_dir)
    clear_directory(val_dir)
    clear_directory(test_dir)

    # Find all .pkl files in the folder
    all_files = glob.glob(os.path.join(folder_path, "*.pkl"))
    random.shuffle(all_files) # Shuffle for random split

    # Calculate split indices
    train_end = int(len(all_files) * train_split)
    val_end = train_end + int(len(all_files) * val_split)

    # Split files
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    # Copy files to respective directories
    for file in train_files:
        shutil.copy(file, train_dir)
    for file in val_files:
        shutil.copy(file, val_dir)
    for file in test_files:
        shutil.copy(file, test_dir)


# Set the base directory using relative paths
base_dir = os.path.join("../../data/graphs/batches")

# List of new folders to process
folders = ["dim_reduced_10", "dim_reduced_100", "gnn_only", "llm_embeddings_full"]

# Split ratios
train_split = 0.8
val_split = 0.1
test_split = 0.1

# Process each folder
create_splits_for_folder("dim_reduced_10")
# for folder in folders:
    # create_splits_for_folder(folder)
