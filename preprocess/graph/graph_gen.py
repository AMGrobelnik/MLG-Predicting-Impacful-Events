import os
import pickle
from typing import List

import networkx as nx
import pandas as pd
import torch
from tqdm import tqdm


def get_file_names(count: int, directory="../../data/preprocessed/") -> List[str]:
    """
    Returns a list of paths to the preprocessed .pkl files
    """
    files = []
    count = min(count, 2944)  # cap the number of files
    for i in range(1, count + 1):
        file = os.path.join(directory, f"events-{str(i).zfill(5)}.pkl")
        file = os.path.abspath(file)
        files.append(file)
    return files


def get_event_attributes(
    event: dict, is_similar: bool, df_llm: pd.DataFrame = None
) -> dict:
    """
    Generates a dictionary of features for a given event
    :param df_llm: DataFrame containing the LLM embeddings
    :param event: the event to generate features for
    :param is_similar: whether the event is a similar event (and does not have all the attributes)
    """
    global remove_future, count_feature

    uri = event["info"]["uri"] if not is_similar else event["uri"]
    article_count = -1 if is_similar else event["info"]["articleCounts"]["total"]
    event_date = event["eventDate"] if is_similar else event["info"]["eventDate"]

    feature_tensor = torch.tensor([event_date], dtype=torch.float32)
    if df_llm is not None:
        if uri in df_llm.index:
            llm = df_llm.loc[uri]
            # title = torch.tensor(llm["title"], dtype=torch.float32)
            summary = torch.tensor(llm["summary"], dtype=torch.float32)
            feature_tensor = torch.cat((feature_tensor, summary))  # title
        else:
            if not is_similar:
                tqdm.write(f"WARNING: LLM not found for {uri}")
            feature_tensor = torch.cat((feature_tensor, torch.zeros(embedding_dim * 1)))

    if count_feature:
        feature_tensor = torch.cat(
            (torch.tensor([article_count], dtype=torch.float32), feature_tensor)
        )

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
        (event_id, c["id"], {"edge_type": "related"})  # , "weight": c["score"]
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
        (event_id, se["uri"], {"edge_type": "similar"})  # , "weight": se["sim"]
        for se in similar_events
    ]

    return nodes, edges


def load_llm_embeddings(file: str):
    file_name = file.split("/")[-1]
    df = pd.read_pickle(f"../../data/text/{embedded_directory}/{file_name}")
    return df


def generate_graph(
    n_files: int,
    include_concepts: bool = True,
    include_similar_events: bool = True,
    include_llm_embeddings: bool = True,
    no_unknown: bool = True,
) -> nx.Graph:
    """
    Generates a graph from a list of files
    :param n_files: the number of files to include
    :param include_concepts: include the concepts in the graph
    :param include_similar_events: include the similar events and their edges in the graph
    :param include_llm_embeddings: include the LLM embeddings in the graph
    :param no_unknown: find events without features in other files
    :return: the generated graph
    """
    n_files = min(n_files, 2944)  # cap the number of files
    all_files = get_file_names(3000)  # get all files
    files, other_files = all_files[:n_files], all_files[n_files:]
    G = nx.Graph()

    # Add similar events first to avoid overwriting
    se_ids = set()
    if include_similar_events:
        for file in tqdm(files, desc="Adding similar events", ncols=100):
            df = pd.read_pickle(file)
            df_llm = load_llm_embeddings(file) if include_llm_embeddings else None
            for event_id, event in df.iterrows():
                info, similar_events = event["info"], event["similarEvents"]

                se_nodes, se_edges = get_similar_events(
                    event_id, similar_events, df_llm
                )
                G.add_nodes_from(se_nodes)
                G.add_edges_from(se_edges)

                for se in se_nodes:
                    se_ids.add(se[0])

    # Add event data
    for file in tqdm(files, desc="Adding event data", ncols=100):
        df = pd.read_pickle(file)
        df_llm = load_llm_embeddings(file) if include_llm_embeddings else None
        for event_id, event in df.iterrows():
            G.add_node(event_id, **get_event_attributes(event, False, df_llm))

            # remove id from similar events
            if event_id in se_ids:
                se_ids.remove(event_id)

    # Add concepts
    if include_concepts:
        for file in tqdm(files, desc="Adding concepts", ncols=100):
            df = pd.read_pickle(file)
            for event_id, event in df.iterrows():
                info = event["info"]
                c_nodes, c_edges = get_concepts(event_id, info["concepts"])
                G.add_nodes_from(c_nodes)
                G.add_edges_from(c_edges)

    if not no_unknown:
        return G

    # Add data from other files to similar events
    for file in tqdm(other_files, desc="Adding data from other files", ncols=100):
        if len(se_ids) == 0:
            break

        df = pd.read_pickle(file)

        # Find the ids in the file
        df_ids = set(df.index)
        ids = df_ids.intersection(se_ids)
        if len(ids) == 0:
            continue

        # remove the ids from the list
        se_ids = se_ids.difference(ids)

        df_llm = load_llm_embeddings(file) if include_llm_embeddings else None
        for event_id in ids:
            event = df.loc[event_id]
            G.add_node(event_id, **get_event_attributes(event, False, df_llm))

            if include_concepts:
                c_nodes, c_edges = get_concepts(event_id, event["info"]["concepts"])
                G.add_nodes_from(c_nodes)
                G.add_edges_from(c_edges)
    return G


def save_graph(graph: nx.Graph, name: str, directory="../../data/graphs/"):
    """
    Saves the graph in a pickle file
    """
    path = os.path.join(directory, f"{name}.pkl")
    tqdm.write(f"Saving to {name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def add_concept_features(graph: nx.Graph, llm_embeddings: bool):
    """
    Adds node degree and llm embeddings to the concepts
    """
    if llm_embeddings:
        print("Loading LLM concept embeddings")
        c_ids = pickle.load(open("../../data/text/concept_embeds/c_ids.pkl", "rb"))
        # ^ a dict of {file: indices}

        # iterate concept nodes in the graph and find the corresponding embedding file
        file_to_ids = {}
        concept_ids = set([n for n in graph.nodes() if n.startswith("c")])
        for file in c_ids.keys():
            ids = c_ids[file].intersection(concept_ids)
            concept_ids = concept_ids.difference(ids)

            if len(ids) > 0:
                file_to_ids[file] = ids

        # add features to the graph, one embedding file at a time
        for file in tqdm(file_to_ids.keys(), desc="Iterating concept files", ncols=100):
            embeds = pickle.load(
                open(f"../../data/text/{concept_embeds_filename}/{file}", "rb")
            )
            for node in file_to_ids[file]:
                degree = graph.degree(node)
                llm = torch.tensor(embeds.loc[node]["label"], dtype=torch.float32)
                features = torch.tensor([degree], dtype=torch.float32)
                features = torch.cat((features, llm))
                graph.nodes[node]["node_feature"] = features

    else:
        for node in tqdm(graph.nodes(), desc="Adding concept feats", ncols=100):
            if graph.nodes[node]["node_type"] != "concept":
                continue
            degree = graph.degree(node)
            features = torch.tensor([degree], dtype=torch.float32)
            graph.nodes[node]["node_feature"] = features

    # add self loops to concepts
    for node in tqdm(graph.nodes(), desc="Adding self loops", ncols=100):
        if graph.nodes[node]["node_type"] == "concept":
            graph.add_edge(node, node, edge_type="related")


def prune_disconnected(graph: nx.Graph):
    """
    Removes all nodes that are not connected to any other node
    """
    isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(isolates)
    tqdm.write(f"Removed {len(isolates)} disconnected nodes")


def remove_future_edges(graph: nx.Graph, threshold: int):
    """
    Removes all edges that point to the future
    """
    to_remove = []
    for u, v, data in tqdm(graph.edges(data=True), desc="Removing future", ncols=100):
        if data["edge_type"] == "related":
            # remove event -> concept edges
            if u.startswith("e"):
                to_remove.append((u, v))
        else:
            d1 = graph.nodes[u]["node_feature"][0]
            d2 = graph.nodes[v]["node_feature"][0]

            # remove the edge if
            # - it points to the past
            # - they happen at the same time (i.e. within the threshold)
            if d1 > d2 or abs(d1 - d2) <= threshold:
                to_remove.append((u, v))

    graph.remove_edges_from(to_remove)
    tqdm.write(f"Removed {len(to_remove)} future edges")


def remove_unknown_events(graph: nx.Graph):
    """
    Removes all events that do not have a target
    """
    to_remove = []
    for node, data in tqdm(
        graph.nodes(data=True), desc="Removing unknown events", ncols=100
    ):
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
    concept_edges = len(
        [
            data
            for _, _, data in graph.edges(data=True)
            if data["edge_type"] == "related"
        ]
    )
    event_edges = len(
        [
            data
            for _, _, data in graph.edges(data=True)
            if data["edge_type"] == "similar"
        ]
    )
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


def get_referenced_ids(n_files: int):
    """
    Returns a list of all the ids that are referenced in the files
    """
    files = get_file_names(n_files)

    e_ids = set()
    c_ids = set()

    for file in tqdm(files, desc="Getting referenced ids", ncols=100):
        df = pd.read_pickle(file)
        for _, event in df.iterrows():
            e_ids.add(event["info"]["uri"])
            for c in event["info"]["concepts"]:
                c_ids.add(c["id"])
            for se in event["similarEvents"]:
                e_ids.add(se["uri"])

    return e_ids, c_ids


n = 1
concepts = True
similar = True
llm_embeddings = True

remove_isolates = True
remove_future = True
future_threshold = 2
no_unknown = True
count_feature = False  # include article counts in the node features

embedded_directory = "embedded"
concept_embeds_filename = "concept_embeds"
embedding_dim = 768


def main():
    start_time = pd.Timestamp.now()
    graph = generate_graph(n, concepts, similar, llm_embeddings, no_unknown)

    # if remove_unknown:
    #     remove_unknown_events(graph)

    tqdm.write("Converting to directed graph")
    graph = nx.DiGraph(graph)

    if remove_future:
        remove_future_edges(graph, future_threshold)
    if remove_isolates:
        prune_disconnected(graph)

    if concepts:
        add_concept_features(graph, llm_embeddings)

    name = f"{n}"
    name += "_concepts" if concepts else ""
    name += "_similar" if similar else ""
    name += "_llm" if llm_embeddings else ""
    name += "_noUnknown" if no_unknown else ""
    name += f"_noFutureThr{future_threshold}" if remove_future else ""
    name += "_noIsolates" if remove_isolates else ""
    name += "_withCounts" if count_feature else ""
    save_graph(graph, name)

    end_time = pd.Timestamp.now()
    print(f"Time taken: {round((end_time - start_time).seconds / 60, 2)} min")
    print_graph_statistics(graph)


if __name__ == "__main__":
    main()
