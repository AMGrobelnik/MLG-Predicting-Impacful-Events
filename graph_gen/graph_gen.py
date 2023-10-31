import os
import pickle
from typing import List

import networkx as nx
import pandas as pd


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

    if similar_event:
        return {"type": "event", "features": {"eventDate": event["eventDate"]}}

    info = event["info"]

    return {
        "type": "event",
        "features": {
            "eventDate": info["eventDate"],
            "article_count": info["articleCounts"]["total"],
            # TODO: LLM embeddings
        },
    }


def get_concept_attributes(c: dict) -> dict:
    """
    Generates a dictionary of features for a given concept
    """
    return {
        "type": "concept",
        "features": {
            "concept_label": c["labelEng"],
            # 'concept_type': c['type'],
            # 'uri': c['uri']
        },
    }


def generate_id(orig_id: str, prefix: str) -> str:
    """
    Generates an ID for a given concept or event
    :param orig_id: the original ID of the concept or event
    :param prefix: the prefix to be added to the ID (e.g. 'c' for concept, 'e' for event)
    """
    return f"{prefix}_{orig_id}"


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

    for file in files:
        df = pd.read_pickle(file)
        for i, event in df.iterrows():
            info, similar_events = event["info"], event["similarEvents"]
            event_id = generate_id(info["uri"], "e")

            G.add_node(event_id, **get_event_attributes(event, False))

            if include_concepts:
                concepts = info["concepts"]
                for concept in concepts:
                    concept_id = generate_id(concept["id"], "c")
                    G.add_node(concept_id, **get_concept_attributes(concept))
                    G.add_edge(
                        event_id,
                        concept_id,
                        edge_type="concept",
                        weight=concept["score"],
                    )

            if include_similar_events:
                for similar_event in similar_events:
                    similar_id = generate_id(similar_event["uri"], "e")
                    G.add_node(similar_id, **get_event_attributes(similar_event, True))
                    G.add_edge(
                        event_id,
                        similar_id,
                        edge_type="similar",
                        weight=similar_event["sim"],
                    )

    return G


def save_graph(graph: nx.Graph, name: str, directory="../data/graphs/"):
    """
    Saves the graph in a pickle file
    """
    path = os.path.join(directory, f"{name}.pkl")

    with open(path, "wb") as f:
        pickle.dump(graph, f)


if __name__ == "__main__":
    # Generate graph
    files = get_file_names(100)

    print("Generating graph...")
    G = generate_graph(files, include_concepts=True, include_similar_events=True)

    print("Saving graph...")
    save_graph(G, "100_concepts_similar")
