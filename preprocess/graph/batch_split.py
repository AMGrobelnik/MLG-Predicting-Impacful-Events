import os
import shutil
import glob
import random

# Navigate two levels up from the current script's directory
base_path = os.path.dirname(os.path.dirname(os.getcwd()))

# Set the directory paths
base_dir = os.path.join(base_path, "data/graphs/batches")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Split ratios
train_split = 0.8
val_split = 0.1
test_split = 0.1

def clear_directory(directory):
    """Delete all .pkl files in the specified directory"""
    for file in glob.glob(os.path.join(directory, "*.pkl")):
        os.remove(file)

def split_files():
    # Clear existing files in the directories
    clear_directory(train_dir)
    clear_directory(val_dir)
    clear_directory(test_dir)

    # Find all .pkl files in the base directory
    all_files = glob.glob(os.path.join(base_dir, "*.pkl"))
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

# Run the file splitting function
split_files()
