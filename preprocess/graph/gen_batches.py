import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from glob import glob
import torch
from deepsnap.hetero_graph import HeteroGraph


def load_data():
    with open("../../data/batch/B_recent_10_khops_2k.pkl", "rb") as f:
        subgraph_ids = pickle.load(f)
        for i in range(len(subgraph_ids)):
            subgraph_ids[i] = (subgraph_ids[i][0], list(subgraph_ids[i][1]))

    with open("../../data/preprocessed/event_index.pkl", "rb") as f:
        event_index = pickle.load(f)
        event_index = reverse_event_index(event_index)
    return subgraph_ids, event_index


def reverse_event_index(event_index):
    new_index = {}
    for file, ids in event_index.items():
        for eid in ids:
            new_index[eid] = file

    return new_index


def save_batch(graphs, start_index):
    for i, graph in enumerate(graphs):
        file_name = f"batch_{str(start_index + i).zfill(5)}.pkl"
        with open("../../data/graphs/batches/" + file_name, "wb") as f:
            pickle.dump(graph, f)


def batch_generate(subgraph_ids, event_index, batch_size):
    """
    Generates subgraphs for training
    :param subgraph_ids: list of tuples (target_ids, neighbor_ids)
    :param event_index: maps event ids -> file names
    :param batch_size: number of graphs to construct at the same time
    :return:
    """

    # split subgraph_ids into batches
    batched_ids = [
        subgraph_ids[i : i + batch_size]
        for i in range(0, len(subgraph_ids), batch_size)
    ]

    # generate batches
    i = 0
    for batch in tqdm(batched_ids, desc="Generating batches"):
        graphs = []

        for target_ids, neighbor_ids in batch:
            tqdm.write("Generating subgraph...")
            graph = generate_subgraph(target_ids, neighbor_ids, event_index)
            graphs.append(graph)

            tqdm.write("Converting to deepsnap...")
            graph = HeteroGraph(graph, netlib=nx, directed=True)
            graph['node_target'] = graph._get_node_attributes('node_target')

        # save batch
        tqdm.write(f"Saving batch {i}...")
        save_batch(graphs, i)
        i += batch_size


def get_files_to_idx(t_ids, n_ids, event_index):
    """
    Maps event ids to file names
    :param t_ids: list of target event ids
    :param n_ids: list of neighbor event ids
    :param event_index: maps event ids -> file names
    :return:
    """
    file_to_idx = {}
    filtered_ids = []
    for ids in [t_ids, n_ids]:
        for eid in ids:
            # skip events not in the dataset
            if eid not in event_index:
                continue

            file_name = event_index[eid]
            if file_name not in file_to_idx:
                file_to_idx[file_name] = set()

            file_to_idx[file_name].add(eid)
            filtered_ids.append(eid)

    return file_to_idx, set(filtered_ids)


def load_concept_llm_files():
    files = glob("../../data/text/concept_embeds/concept_embeds_*.pkl")
    llm_files = {}
    for file in files:
        with open(file, "rb") as f:
            llm_file = pickle.load(f)
            llm_files[file] = llm_file

    # with open(f"../../data/text/embedded/{file_name}.pkl", "rb") as f:
    #     file = pickle.load(f)
    #     llm_files[file_name] = file

    return llm_files


src_files = None


def load_files(files_to_idx):
    global src_files
    """
    Loads files into memory
    :param files_to_idx: maps file names -> event ids
    :return:
    """
    if src_files is None:
        src_files = {}

    for file_name in files_to_idx.keys():
        if file_name in src_files:
            continue
        with open(f"../../data/preprocessed_dicts/f{file_name}.pkl", "rb") as f:
            file = pickle.load(f)
            src_files[file_name] = file

    return src_files


def add_event(graph, event_id, e_type, all_nodes, src_file):
    """
    Adds an event to the graph
    :param graph:
    :param event_id: event id to add
    :param e_type: type of event ('event' or 'event_target')
    :param all_nodes: list of all nodes in the graph
    :param src_file: source file
    :return:
    """

    event = src_file[event_id]
    info = event["info"]
    event_counts = info["articleCounts"]["total"]
    event_date = info["eventDate"]
    concepts = info["concepts"]
    similar = event["similar_events"]

    features = np.array([event_date], dtype=np.float32)
    target = None

    if e_type == "event":
        features = np.concatenate([np.array([event_counts]), features], dtype=np.float32)
    else:
        target = torch.tensor([event_counts], dtype=torch.float32)

    features = torch.from_numpy(features)

    # add node
    if e_type == "event":
        graph.add_node(event_id, node_type=e_type, node_feature=features)
    else:
        graph.add_node(event_id, node_type=e_type, node_feature=features, node_target=target)

    # add similar event edges
    for se in similar:
        se_id = se["uri"]
        se_date = se["eventDate"]
        if se_id not in all_nodes:
            continue

        e_from, e_to = event_id, se_id
        if se_date < event_date:
            e_from, e_to = e_to, e_from

        # no edges of type (event_target, similar, event)
        if e_from == event_id and e_type == "event_target":
            continue

        graph.add_edge(e_from, e_to, edge_type="similar")

    # add concepts
    for concept in concepts:
        concept_id = concept["id"]

        graph.add_node(concept_id, node_type="concept")
        graph.add_edge(concept_id, event_id, edge_type="related")
        graph.add_edge(concept_id, concept_id, edge_type="concept_self")


def generate_subgraph(target_ids, neighbor_ids, event_index):
    """
    Generates a subgraph for training
    :param target_ids: list of target ids
    :param neighbor_ids: list of neighbor ids
    :param event_index: maps event ids -> file names
    :return:
    """
    # get files to idx
    files_to_idx, all_nodes = get_files_to_idx(target_ids, neighbor_ids, event_index)
    src_files = load_files(files_to_idx)

    graph = nx.DiGraph()
    for file_name, ids in files_to_idx.items():
        src_file = src_files[file_name]

        for eid in ids:
            event_type = "event_target" if eid in target_ids else "event"
            add_event(graph, eid, event_type, all_nodes, src_file)

    # add degree to concepts
    for node in graph.nodes():
        if graph.nodes[node]["node_type"] == "concept":
            graph.nodes[node]["node_feature"] = torch.tensor(
                [graph.degree[node] - 1], dtype=torch.float32
            )

    return graph


def main():
    subgraph_ids, event_index = load_data()

    subgraph_ids = subgraph_ids[0]

    batch_generate(subgraph_ids, event_index, 10)


if __name__ == "__main__":
    main()
