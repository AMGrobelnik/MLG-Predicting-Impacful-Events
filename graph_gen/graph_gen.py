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


def get_event_attributes(
    event: dict, similar_event: bool, df_llm: pd.DataFrame = None
) -> dict:
    """
    Generates a dictionary of features for a given event
    :param df_llm: DataFrame containing the LLM embeddings
    :param event: the event to generate features for
    :param similar_event: whether the event is a similar event (and does not have all the attributes)
    """
    global remove_future, cheat

    uri = event["info"]["uri"] if not similar_event else event["uri"]
    article_count = -1 if similar_event else event["info"]["articleCounts"]["total"]
    event_date = event["eventDate"] if similar_event else event["info"]["eventDate"]

    feature_tensor = torch.tensor([event_date], dtype=torch.float32)
    if df_llm is not None:
        if uri in df_llm.index:
            llm = df_llm.loc[uri]
            # title = torch.tensor(llm["title_embed"], dtype=torch.float32)
            summary = torch.tensor(llm["summary_embed"], dtype=torch.float32)
            feature_tensor = torch.cat((feature_tensor, summary))  # title
        else:
            feature_tensor = torch.cat((feature_tensor, torch.zeros(768 * 1)))

    if remove_future or cheat:  # TODO: hide the article_count from train data
        feature_tensor = torch.cat((torch.tensor([article_count], dtype=torch.float32), feature_tensor))

    return {
        "node_type": "event",
        "node_feature": feature_tensor,
        "node_target": torch.tensor([article_count], dtype=torch.float32),
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
        "node_feature": None,  # Added later
    }


def get_concepts(event_id: str, concepts: dict):
    nodes = [(c["id"], get_concept_attributes(c)) for c in concepts]
    edges = [
        (event_id, c["id"], {"edge_type": "related", "weight": c["score"]})
        for c in concepts
    ]

    return nodes, edges


def get_similar_events(
    event_id: str, similar_events: dict, llm_df: pd.DataFrame = None
):
    nodes = [
        (se["uri"], get_event_attributes(se, True, llm_df)) for se in similar_events
    ]
    edges = [
        (event_id, se["uri"], {"edge_type": "similar", "weight": se["sim"]})
        for se in similar_events
    ]

    return nodes, edges


def generate_graph(
    files: List[str],
    include_concepts: bool = True,
    include_similar_events: bool = True,
    include_llm_embeddings: bool = True,
) -> nx.Graph:
    """
    Generates a graph from a list of files
    :param files: list of file paths
    :param include_concepts: include the concepts in the graph
    :param include_similar_events: include the similar events and their edges in the graph
    :return:
    """
    G = nx.Graph()
    df_llm = None
    if include_llm_embeddings:
        df_llm = pd.read_pickle("../data/text/llm_embeddings.pkl")

    for file in tqdm(files, desc="Adding similar events and concepts", ncols=100):
        if not include_similar_events and not include_concepts:
            break
        df = pd.read_pickle(file)
        for i, event in df.iterrows():
            info, similar_events = event["info"], event["similarEvents"]
            event_id = info["uri"]

            if include_concepts:
                c_nodes, c_edges = get_concepts(event_id, info["concepts"])
                G.add_nodes_from(c_nodes)
                G.add_edges_from(c_edges)

            if include_similar_events:
                se_nodes, se_edges = get_similar_events(
                    event_id, similar_events, df_llm
                )
                G.add_nodes_from(se_nodes)
                G.add_edges_from(se_edges)

    # Add event data last to avoid overwriting
    for file in tqdm(files, desc="Adding event data", ncols=100):
        df = pd.read_pickle(file)
        for _, event in df.iterrows():
            info, similar_events = event["info"], event["similarEvents"]
            event_id = info["uri"]
            G.add_node(event_id, **get_event_attributes(event, False, df_llm))

    return G


def save_graph(graph: nx.Graph, name: str, directory="../data/graphs/"):
    """
    Saves the graph in a pickle file
    """
    path = os.path.join(directory, f"{name}.pkl")
    tqdm.write(f"Saving to {name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def add_concept_degree(graph: nx.Graph):
    """
    Adds node degree as a feature to concepts
    """
    for node in tqdm(graph.nodes(), desc="Adding node degree to concepts", ncols=100):
        if graph.nodes[node]["node_type"] != "concept":
            continue
        graph.nodes[node]["node_feature"] = torch.tensor(
            [graph.degree(node)], dtype=torch.float32
        )


def prune_disconnected(graph: nx.Graph):
    """
    Removes all nodes that are not connected to any other node
    """
    nodes_to_remove = []
    for node in tqdm(graph.nodes(), desc="Removing disconnected nodes", ncols=100):
        if graph.degree(node) == 0:
            nodes_to_remove.append(node)
    graph.remove_nodes_from(nodes_to_remove)
    tqdm.write(f"Removed {len(nodes_to_remove)} disconnected nodes")


def remove_future_edges(graph: nx.Graph, threshold: int):
    """
    Removes all edges that point to the future
    """
    to_remove = []
    for u, v, data in tqdm(graph.edges(data=True), desc="Removing future", ncols=100):
        if data["edge_type"] != "similar":
            continue
        d1 = graph.nodes[u]["node_feature"][0]
        d2 = graph.nodes[v]["node_feature"][0]

        # simultaneous events are not removed
        if abs(d1 - d2) <= threshold:
            continue

        # remove the edge if it points to the past
        if d1 < d2:
            to_remove.append((u, v))

    graph.remove_edges_from(to_remove)
    tqdm.write(f"Removed {len(to_remove)} future edges")


def remove_unknown_events(graph: nx.Graph):
    """
    Removes all events that do not have a target
    """
    to_remove = []
    for node, data in tqdm(graph.nodes(data=True), desc="Removing unknown events", ncols=100):
        if data["node_type"] != "event":
            continue
        if data["node_target"][0] == -1:
            to_remove.append(node)
    graph.remove_nodes_from(to_remove)
    tqdm.write(f"Removed {len(to_remove)} unknown events")


def print_graph_statistics(graph: nx.Graph):
    """
    Prints some statistics about the graph
    """
    node_c = graph.number_of_nodes()
    edge_c = graph.number_of_edges()
    concepts = len([n for n in graph.nodes() if n.startswith("c")])
    events = len([n for n in graph.nodes() if n.startswith("e")])
    concept_edges = len([data for _, _, data in graph.edges(data=True) if data["edge_type"] == "related"])
    event_edges = len([data for _, _, data in graph.edges(data=True) if data["edge_type"] == "similar"])
    counts = 0
    for _, data in graph.nodes(data=True):
        if data["node_type"] == "event":
            if data["node_target"][0] > 0:
                counts += 1

    tqdm.write("Success!!")
    tqdm.write(f"Number of nodes: {node_c}")
    tqdm.write(f"Number of edges: {edge_c}")
    tqdm.write(f"Number of concepts: {concepts} ({round(concepts / node_c * 100, 2)}%)")
    tqdm.write(f"Number of events: {events} ({round(events / node_c * 100, 2)}%)")
    tqdm.write(
        f"Number of events with target: {counts} ({round(counts / events * 100, 2)}%)"
    )
    tqdm.write(
        f"Number of concept edges: {concept_edges} ({round(concept_edges / edge_c * 100, 2)}%)"
    )
    tqdm.write(
        f"Number of event edges: {event_edges} ({round(event_edges / edge_c * 100, 2)}%)"
    )


n = 10
concepts = True
similar = True
llm_embeddings = True
cheat = True  # include article counts in the node features

remove_unknown = True
remove_future = False
future_threshold = 0


if __name__ == "__main__":
    files = get_file_names(n)

    graph = generate_graph(files, concepts, similar, llm_embeddings)

    if remove_unknown:
        remove_unknown_events(graph)  # TODO: Maybe keep the edges

    for i in tqdm([1], desc="To directed", ncols=100):
        graph = graph.to_directed()

    if remove_future:
        remove_future_edges(graph, future_threshold)

    prune_disconnected(graph)
    add_concept_degree(graph)

    for i in tqdm([1], desc="Saving graph", ncols=100):
        name = f"{n}"
        name += "_concepts" if concepts else ""
        name += "_similar" if similar else ""
        name += "_llm" if llm_embeddings else ""
        name += "_noFuture" if remove_future else ""
        name += "_noUnknown" if remove_unknown else ""
        name += "_CHEAT" if cheat else ""
        save_graph(graph, name)

    print_graph_statistics(graph)
