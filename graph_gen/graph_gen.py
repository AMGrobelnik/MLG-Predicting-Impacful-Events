import os
import pickle
from typing import List

import networkx as nx
import pandas as pd
import torch
from tqdm import tqdm


def get_file_names(count: int, directory="../data/preprocessed/") -> List[str]:
    """
    Returns a list of paths to the preprocessed .pkl files
    """
    files = []
    for i in range(1, count + 1):
        file = os.path.join(directory, f"events-{str(i).zfill(5)}.pkl")
        file = os.path.abspath(file)
        files.append(file)
    return files


def get_event_attributes(event: dict, similar_event: bool) -> dict:
    """
    Generates a dictionary of features for a given event
    :param event: the event to generate features for
    :param similar_event: whether the event is a similar event (and does not have all the attributes)
    """
    article_count = -1 if similar_event else event["info"]["articleCounts"]["total"]
    event_date = event["eventDate"] if similar_event else event["info"]["eventDate"]
    feature_tensor = torch.tensor([article_count, event_date])

    return {
        "node_type": "event",
        "node_feature": feature_tensor,
        # TODO: LLM embeddings
    }


def get_concept_attributes(c: dict) -> dict:
    """
    Generates a dictionary of features for a given concept
    """
    return {
        "node_type": "concept",
        # "concept_label": c["labelEng"],
        # 'concept_type': c['type'],
        # 'uri': c['uri']
        "node_feature": torch.tensor([1, 1, 1, 1, 1]),
    }


def get_concepts(event_id: str, concepts: dict):
    nodes = [(c["id"], get_concept_attributes(c)) for c in concepts]
    edges = [
        (event_id, c["id"], {"edge_type": "related", "weight": c["score"]})
        for c in concepts
    ]

    return nodes, edges


def get_similar_events(event_id: str, similar_events: dict):
    nodes = [(se["uri"], get_event_attributes(se, True)) for se in similar_events]
    edges = [
        (event_id, se["uri"], {"edge_type": "similar", "weight": se["sim"]})
        for se in similar_events
    ]

    return nodes, edges


def generate_graph(
    files: List[str],
    include_concepts: bool = False,
    include_similar_events: bool = True,
) -> nx.Graph:
    """
    Generates a graph from a list of files
    :param files: list of file paths
    :param include_concepts: include the concepts in the graph
    :param include_similar_events: include the similar events and their edges in the graph
    :return:
    """
    G = nx.Graph()

    for file in tqdm(files, desc="Generating graph", ncols=100):
        df = pd.read_pickle(file)
        for i, event in df.iterrows():
            info, similar_events = event["info"], event["similarEvents"]
            event_id = info["uri"]
            G.add_node(event_id, **get_event_attributes(event, False))

            if include_concepts:
                c_nodes, c_edges = get_concepts(event_id, info["concepts"])
                G.add_nodes_from(c_nodes)
                G.add_edges_from(c_edges)

            if include_similar_events:
                se_nodes, se_edges = get_similar_events(event_id, similar_events)
                G.add_nodes_from(se_nodes)
                G.add_edges_from(se_edges)

    return G


def save_graph(graph: nx.Graph, name: str, directory="../data/graphs/"):
    """
    Saves the graph in a pickle file
    """
    path = os.path.join(directory, f"{name}.pkl")

    with open(path, "wb") as f:
        pickle.dump(graph, f)


if __name__ == "__main__":
    n = 1
    concepts = False
    similar = True

    files = get_file_names(n)

    graph = generate_graph(
        files, include_concepts=concepts, include_similar_events=similar
    )

    print("Saving graph...")
    save_graph(
        graph, f"{n}{'_concepts' if concepts else ''}{'_similar' if similar else ''}"
    )
