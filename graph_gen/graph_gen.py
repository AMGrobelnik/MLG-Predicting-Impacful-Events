import os
import pickle
from typing import List

import networkx as nx
import pandas as pd
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
                G.add_nodes_from(
                    [(c["id"], get_concept_attributes(c)) for c in info["concepts"]]
                )
                G.add_edges_from(
                    [
                        (
                            event_id,
                            c["id"],
                            {"edge_type": "concept", "weight": c["score"]},
                        )
                        for c in info["concepts"]
                    ]
                )

            if include_similar_events:
                G.add_nodes_from(
                    [
                        (se["uri"], get_event_attributes(se, True))
                        for se in similar_events
                    ]
                )
                G.add_edges_from(
                    [
                        (
                            event_id,
                            se["uri"],
                            {"edge_type": "similar", "weight": se["sim"]},
                        )
                        for se in similar_events
                    ]
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
    n = 1000
    concepts = False
    similar = True

    files = get_file_names(n)

    G = generate_graph(files, include_concepts=concepts, include_similar_events=similar)

    print("Saving graph...")
    save_graph(
        G, f"{n}{'_concepts' if concepts else ''}{'_similar' if similar else ''}"
    )
