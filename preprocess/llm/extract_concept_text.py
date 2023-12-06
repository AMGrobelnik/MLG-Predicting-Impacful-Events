import os
import pandas as pd
from tqdm import tqdm

from preprocess.preprocess import save_df_to_pickle

ids = set()
errs = 0
def extract_text(file_path: str):
    global ids, errs
    with open(file_path, "rb") as f:
        df = pd.read_pickle(f)

    data = pd.DataFrame(columns=['id', 'label'])
    data.set_index("id", inplace=True)

    for i, event in df.iterrows():
        try:
            concepts = event['info']["concepts"]
            for concept in concepts:
                c_id = concept['id']
                if c_id not in ids:
                    ids.add(c_id)
                    data.loc[c_id] = [concept['labelEng']]
        except:
            errs += 1
            continue

    return data


if __name__ == "__main__":
    directory_path = "../../data/preprocessed"
    output_dir = "../../data/text"

    files = sorted(os.listdir(directory_path))
    files = [filename for filename in files if filename.endswith(".pkl")]

    data = pd.DataFrame(columns=['id', 'label'])
    data.set_index("id", inplace=True)

    for filename in tqdm(files, ncols=100, desc="Processing"):
        file_path = os.path.join(directory_path, filename)
        file_name = os.path.splitext(filename)[0]

        df = extract_text(file_path)
        data = pd.concat([data, df])

        # save_df_to_pickle(df, output_dir, file_name)

    save_df_to_pickle(data, output_dir, "concepts_text")
    print(f"errs: {errs}")
    print(f"total: {len(ids)}")
    print(data.head())
    print(data.shape)
    print(data.info())