import requests
from tqdm import tqdm
import os


if __name__ == "__main__":
    link = ''

    if not os.path.exists("./data/download"):
        os.makedirs("./data/download")

    r = requests.get(link, stream=True, timeout=10)
    total_size = int(r.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open("./data/download/EventRegistryEvents-July2015.rar", "wb") as f:
        for data in r.iter_content(1024):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
