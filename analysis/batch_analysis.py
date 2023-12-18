import pandas as pd
import numpy as np
import pickle
from umap import UMAP
from tqdm import tqdm


def load_batches(path):
    with open(path, "rb") as f:
        graphs = pickle.load(f)
    return graphs


def days_timestamp_to_date(days):
    return pd.to_datetime(days, unit='D', origin=pd.Timestamp('1970-01-01'))


def get_dataframe(graph):
    # Define the data types for each column
    dtypes = {
        "node_id": "str",
        "node_type": "str",
        "event_date": "str",
        "article_count": "float32",
        "prediction": "float32",
        "l1": "float32",
        "mape": "float32",
    }

    nodes = []
    embeddings = []
    for node, data in tqdm(graph.nodes(data=True), desc="Adding nodes to dataframe"):
        node_type = data["node_type"]
        embedding = data["embedding"]
        embeddings.append(embedding)

        temp_dict = {
            "node_id": node,
            "node_type": node_type,
            "event_date": None,
            "article_count": None,
            "prediction": None,
            'l1': None,
            'mape': None,
        }

        if node_type == "concept":
            nodes.append(temp_dict)
            continue

        features = data["node_feature"]

        if node_type == "event":
            temp_dict["article_count"] = features[0]
            temp_dict["event_date"] = int(features[1] * 16600)
        else:
            temp_dict["event_date"] = int(features[0] * 16600)
            temp_dict["article_count"] = data["node_target"][0]
            temp_dict["prediction"] = data["prediction"][0]
            temp_dict['l1'] = abs(temp_dict['article_count'] - temp_dict['prediction'])
            temp_dict['mape'] = abs(temp_dict['article_count'] - temp_dict['prediction']) / temp_dict['article_count']

        temp_dict['event_date'] = days_timestamp_to_date(temp_dict['event_date'])

        nodes.append(temp_dict)

    node_df = pd.DataFrame(nodes)
    node_df = node_df.astype(dtypes)
    node_df.set_index("node_id", inplace=True)

    embeds = np.array(embeddings)
    return node_df, embeds


def reverse_event_index(event_index):
    new_index = {}
    for file, ids in event_index.items():
        for eid in ids:
            new_index[eid] = file

    return new_index


def get_event_index():
    with open("../data/preprocessed/event_index.pkl", "rb") as f:
        event_index = pickle.load(f)
        event_index = reverse_event_index(event_index)

    return event_index


def get_src_file(file_name):
    with open(f"../data/preprocessed/{file_name}.pkl", "rb") as f:
        file = pd.read_pickle(f)
    return file


def add_event_data(df):

    event_index = get_event_index()
    files = {}
    for i in df.index:
        if i.startswith("c"):
            continue
        if i not in event_index:
            raise Exception(f"Node {i} not in event index")

        file = event_index[i]
        if file not in files:
            files[file] = set()
        files[file].add(i)

    for file_name, indices in tqdm(files.items(), desc="Loading event files"):
        file = get_src_file(file_name)

        for i in indices:
            event = file.loc[i]
            info = event['info']
            multiling = info['multiLingInfo']
            langs = multiling.keys()
            lang_count = len(langs)

            # add event title, summary, lang
            if "eng" in langs:
                lang = "eng"
                eng = multiling["eng"]
                df.loc[i, "title"] = eng["title"]
                df.loc[i, "summary"] = eng["summary"]
                df.loc[i, "lang"] = "eng"

            elif lang_count > 0:
                lang = list(langs)[0]
            else:
                raise Exception(f"Event {i} has no language")

            lang_info = multiling[lang]
            df.loc[i, "title"] = lang_info["title"]
            df.loc[i, "summary"] = lang_info["summary"]
            df.loc[i, "lang"] = lang

            # add concept titles
            concepts = info['concepts']
            for c in concepts:
                c_id = c['id']
                if c_id not in df.index:
                    continue

                df.loc[c_id, "title"] = c['labelEng']

    # for i, row in tqdm(df.iterrows(), desc="Adding event data"):
    #     if i.startswith("c"):
    #         continue
    #     if i not in event_index:
    #         raise Exception(f"Node {i} not in event index")
    #
    #     file_name = event_index[i]
    #     file = get_src_file(file_name)
    #
    #     event = file.loc[i]
    #     info = event['info']
    #     multiling = info['multiLingInfo']
    #     langs = multiling.keys()
    #     lang_count = len(langs)
    #
    #     # add event title, summary, lang
    #     if "eng" in langs:
    #         lang = "eng"
    #         eng = multiling["eng"]
    #         df.loc[i, "title"] = eng["title"]
    #         df.loc[i, "summary"] = eng["summary"]
    #         df.loc[i, "lang"] = "eng"
    #
    #     elif lang_count > 0:
    #         lang = list(langs)[0]
    #     else:
    #         raise Exception(f"Event {i} has no language")
    #
    #     lang_info = multiling[lang]
    #     df.loc[i, "title"] = lang_info["title"]
    #     df.loc[i, "summary"] = lang_info["summary"]
    #     df.loc[i, "lang"] = lang
    #
    #     # add concept titles
    #     concepts = info['concepts']
    #     for c in concepts:
    #         c_id = c['id']
    #         if c_id not in df.index:
    #             continue
    #
    #         df.loc[c_id, "title"] = c['labelEng']

    return df



def get_data(graph):
    df, embeds = get_dataframe(graph)

    df = add_event_data(df)

    umap = UMAP(n_components=2, n_neighbors=5, min_dist=0.5, metric="cosine")
    umap.fit(embeds)

    df["umap_x"] = umap.embedding_[:, 0]
    df["umap_y"] = umap.embedding_[:, 1]

    return df


def main():
    graphs = load_batches('./graphs.pkl')
    graph = graphs[0]
    df = get_data(graph)



if __name__ == "__main__":
    main()