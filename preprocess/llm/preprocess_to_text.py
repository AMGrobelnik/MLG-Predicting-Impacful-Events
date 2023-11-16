import os

import pandas as pd
from tqdm import tqdm

from preprocess import save_df_to_pickle

columns = ["id", "lang", "title", "summary", "article_count", "event_date"]


def extract_text(file_path: str) -> pd.DataFrame:
    # load df from pickle
    df = pd.read_pickle(file_path)
    data = pd.DataFrame(columns=columns)
    data.set_index("id", inplace=True)

    for i, event in df.iterrows():
        e_id = event["info"]["uri"]
        multiling = event["info"]["multiLingInfo"]
        langs = multiling.keys()
        lang_count = len(langs)
        article_count = event["info"]["articleCounts"]["total"]
        event_date = event['info']['eventDate']

        if "eng" in langs:
            eng = multiling["eng"]
            data.loc[e_id] = [
                "eng",
                eng["title"],
                eng["summary"],
                article_count,
                event_date
            ]

        elif lang_count > 0:
            lang = list(langs)[0]
            lang_info = multiling[lang]
            data.loc[e_id] = [
                lang,
                lang_info["title"],
                lang_info["summary"],
                article_count,
                event_date
            ]
        else:
            data.loc[e_id] = [None, None, None, article_count, event_date]

    return data


if __name__ == "__main__":
    directory_path = "../../data/preprocessed"
    output_dir = "../../data/text"

    files = sorted(os.listdir(directory_path))
    files = [filename for filename in files if filename.endswith(".pkl")]

    data = pd.DataFrame(columns=columns)
    data.set_index("id", inplace=True)

    for filename in tqdm(files, ncols=100, desc="Processing"):
        file_path = os.path.join(directory_path, filename)
        file_name = os.path.splitext(filename)[0]

        df = extract_text(file_path)
        data = pd.concat([data, df])

        # save_df_to_pickle(df, output_dir, file_name)

    save_df_to_pickle(data, output_dir, "dataset")
