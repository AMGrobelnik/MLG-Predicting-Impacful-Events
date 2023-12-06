import pandas as pd
import numpy as np
import umap
import os


event_filepath = "../../data/text/embedded"
event_output = "../../data/text/embedded_umap"
concept_filepath = "../../data/text/concept_embeds"
concept_output = "../../data/text/concept_embeds_umap"


def reduce_event_emb(i):
    df = pd.read_pickle(f"{event_filepath}/events-{i:05}.pkl")

    title_embeddings = np.array(df['title'].tolist())
    summary_embeddings = np.array(df['summary'].tolist())

    # Apply UMAP reduction to title embeddings
    umap_title = umap.UMAP(n_components=100, random_state=42)
    umap_title_result = umap_title.fit_transform(title_embeddings)

    # Apply UMAP reduction to summary embeddings
    umap_summary = umap.UMAP(n_components=100, random_state=42)
    umap_summary_result = umap_summary.fit_transform(summary_embeddings)

    # Add UMAP results as new columns in the DataFrame
    df['title_umap'] = [x for x in umap_title_result]
    df['summary_umap'] = [x for x in umap_summary_result]

    df = df.drop("title", axis=1)
    df = df.drop("summary", axis=1)
    df = df.rename(columns={'title_umap': 'title', 'summary_umap': 'summary'})

    df.to_pickle(f"{event_output}/events-{i:05}.pkl")


def reduce_concept_emb(i):
    df = pd.read_pickle(f"{concept_filepath}/concept_embeds_{i}.pkl")
    print(df)

    concept_embeddings = np.array(df['label'].tolist())

    umap_concept = umap.UMAP(n_components=100, random_state=42, verbose=True)
    umap_concept_result = umap_concept.fit_transform(concept_embeddings)

    df = df.drop("label", axis=1)
    print(umap_concept_result)
    df["label"] = [x for x in umap_concept_result]
    
    df.to_pickle(f"{concept_output}/concept_embeds_{i}.pkl")


def reduce_event_dim():
    i = 1
    while os.path.exists(f"{event_filepath}/events-{i:05}.pkl"):
        reduce_event_emb(i)
        print(f"Processed event: {i}")
        i += 1

def reduce_concept_dim():
    i = 1
    while os.path.exists(f"{concept_filepath}/concept_embeds_{i}.pkl") and i <= 1:
        reduce_concept_emb(i)
        print(f"Processed concept: {i}")
        i += 1

if __name__ == "__main__":
    # reduce_event_dim()
    reduce_concept_dim()