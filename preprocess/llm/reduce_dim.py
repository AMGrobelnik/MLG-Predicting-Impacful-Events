import pandas as pd
import numpy as np
import os
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf
from umap.parametric_umap import load_ParametricUMAP
from random import sample


input_dim = 768
end_dim = 100

event_filepath = "../../data/text/embedded"
event_output = f"../../data/text/embedded_umap_dim{end_dim}"
concept_filepath = "../../data/text/concept_embeds"
concept_output = f"../../data/text/concept_embeds_umap_dim{end_dim}"


def train_umap_model(umap_model):
    all_dfs = []
    for i in sample(range(1, 2700), 100):
        df = pd.read_pickle(f"{event_filepath}/events-{i:05}.pkl")
        all_dfs.append(df["title"])

    for i in sample(range(1, 2700), 100):
        df = pd.read_pickle(f"{event_filepath}/events-{i:05}.pkl")
        all_dfs.append(df["summary"])

    for i in sample(range(12, 13), 1):
        df = pd.read_pickle(f"{concept_filepath}/concept_embeds_{i}.pkl")
        all_dfs.append(df["label"])

    all_dfs = pd.concat(all_dfs, ignore_index=True)
    umap_model.fit(all_dfs.to_list())
    return umap_model


def reduce_event_emb(i, umap_model):
    df = pd.read_pickle(f"{event_filepath}/events-{i:05}.pkl")

    title_embeddings = np.array(df["title"].tolist())
    summary_embeddings = np.array(df["summary"].tolist())

    umap_title_result = umap_model.transform(title_embeddings)
    umap_summary_result = umap_model.transform(summary_embeddings)

    df = df.drop("title", axis=1)
    df = df.drop("summary", axis=1)
    df["title"] = [x for x in umap_title_result]
    df["summary"] = [x for x in umap_summary_result]
    df.to_pickle(f"{event_output}/events-{i:05}.pkl")


def reduce_concept_emb(i, umap_model):
    df = pd.read_pickle(f"{concept_filepath}/concept_embeds_{i}.pkl")

    concept_embeddings = np.array(df["label"].tolist())
    umap_concept_result = umap_model.fit_transform(concept_embeddings)

    df = df.drop("label", axis=1)
    df["label"] = [x for x in umap_concept_result]

    df.to_pickle(f"{concept_output}/concept_embeds_{i}.pkl")


def reduce_event_dim(umap_model):
    if not os.path.exists(event_output):
        os.makedirs(event_output)

    i = 1
    while os.path.exists(f"{event_filepath}/events-{i:05}.pkl"):
        if not os.path.exists(f"{event_output}/events-{i:05}.pkl"):
            reduce_event_emb(i, umap_model)
            print(f"Processed event: {i}")
        i += 1


def reduce_concept_dim(uamp_model):
    if not os.path.exists(concept_output):
        os.makedirs(concept_output)

    i = 1
    while os.path.exists(f"{concept_filepath}/concept_embeds_{i}.pkl"):
        reduce_concept_emb(i, umap_model)
        print(f"Processed concept: {i}")
        i += 1


if __name__ == "__main__":
    umap_model = None
    if os.path.exists(f"best_umap_model_dim{end_dim}"):
        print("Loading model")
        umap_model = load_ParametricUMAP(f"best_umap_model{end_dim}")
    else:
        encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(input_dim,)),
                tf.keras.layers.Dense(units=512, activation="relu"),
                tf.keras.layers.Dense(units=256, activation="relu"),
                tf.keras.layers.Dense(units=128, activation="relu"),
                tf.keras.layers.Dense(units=end_dim),
            ]
        )

        umap_model = ParametricUMAP(
            encoder=encoder,
            n_epochs=50,
            n_components=end_dim,
            random_state=42,
            verbose=True,
        )
        umap_model = train_umap_model(umap_model)
        umap_model.save(f"best_umap_model{end_dim}")
        print("Training done")

    print("Reducing Event Embedding Dimensionality")
    reduce_event_dim(umap_model)
    print("Reducing Concept Embedding Dimensionality")
    reduce_concept_dim(umap_model)
