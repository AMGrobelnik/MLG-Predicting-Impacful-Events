# COLAB INSTALLATION
# ==================

# import os
# import torch
#
# torch_version = str(torch.__version__)
# scatter_src = f"https://pytorch-geometric.com/whl/torch-{torch_version}.html"
# sparse_src = f"https://pytorch-geometric.com/whl/torch-{torch_version}.html"
# !pip install torch-scatter -f $scatter_src
# !pip install torch-sparse -f $sparse_src
# !pip install torch-geometric
# !pip install -q git+https://github.com/snap-stanford/deepsnap.git
# !pip install -U -q PyDrive
# !pip install wandb optuna

# !nvcc --version
# !python -c "import torch; print(torch.version.cuda)"
#
# import torch
# print(torch.__version__)
# import torch_geometric
# print(torch_geometric.__version__)

# ==================


import requests
from tqdm import tqdm
import zipfile
import os


def download_file_from_dropbox(url, destination):
    # Check if the folder already exists
    if os.path.exists(destination):
        print(f"'{destination}' already exists. Download skipped.")
        return

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

        with open(destination, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR, something went wrong")
    else:
        print(f"Failed to download file, status code: {response.status_code}")


def unzip_folder(zip_file, extract_to):
    # Check if the zip file exists
    if not os.path.exists(zip_file):
        print(f"Zip file '{zip_file}' does not exist.")
        return

    # Check if the extraction folder already exists
    if os.path.exists(extract_to):
        print(f"Extraction folder '{extract_to}' already exists. Unzipping skipped.")
        return

    # Create the directory where the contents will be extracted
    os.makedirs(extract_to, exist_ok=True)

    # Extract the contents of the zip file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Files extracted to '{extract_to}'")


def download_and_extract(url, file_name, extract_to):
    download_file_from_dropbox(url, file_name)
    unzip_folder(file_name, extract_to)
