import pickle
import numpy as np
import os
import networkx as nx
from tqdm import tqdm
from glob import glob
from deepsnap.hetero_graph import HeteroGraph
from torch_sparse import SparseTensor
import torch


def load_data():
    with open("../../data/batch/B_recent_10_khops_12k.pkl", "rb") as f:
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


def save_batch(graphs, start_index, hetero):
    global batch_folder

    for i, graph in enumerate(graphs):
        file_name = (
            f"batch{'_hg' if hetero else ''}_{str(start_index + i).zfill(5)}.pkl"
        )
        with open(f"../../data/graphs/{batch_folder}/" + file_name, "wb") as f:
            pickle.dump(graph, f)


def get_hetero_graph(G):
    hete = HeteroGraph(G, directed=True)
    hete["node_target"] = hete._get_node_attributes("node_target")

    for key in hete["node_target"]:
        node_target = hete["node_target"][key]
        node_target = np.array(node_target)
        hete["node_target"][key] = torch.tensor(node_target, dtype=torch.float32)

    for key in hete["node_feature"]:
        node_feature = hete["node_feature"][key]
        node_feature = np.array(node_feature)
        hete["node_feature"][key] = torch.tensor(node_feature, dtype=torch.float32)

    for key in hete.edge_index:
        edge_index = hete.edge_index[key]

        adj = SparseTensor(
            row=edge_index[0].long(),
            col=edge_index[1].long(),
            sparse_sizes=(
                hete.num_nodes(key[0]),
                hete.num_nodes(key[2]),
            ),
        )
        hete.edge_index[key] = adj.t()

    del hete.G

    return hete


def batch_generate(subgraph_ids, event_index, batch_size):
    """
    Generates subgraphs for training
    :param subgraph_ids: list of tuples (target_ids, neighbor_ids)
    :param event_index: maps event ids -> file names
    :param batch_size: number of graphs to construct at the same time
    :return:
    """
    global save_deepsnap, save_nx

    # split subgraph_ids into batches
    batched_ids = [
        subgraph_ids[i : i + batch_size]
        for i in range(0, len(subgraph_ids), batch_size)
    ]

    # generate batches
    i = 0
    for batch in tqdm(batched_ids, desc="Generating batches"):
        graphs = []
        hetero_graphs = []

        for target_ids, neighbor_ids in batch:
            graph = generate_subgraph(target_ids, neighbor_ids, event_index)
            graphs.append(graph)

            if save_deepsnap:
                hg = get_hetero_graph(graph)
                hetero_graphs.append(hg)

        # save batch
        tqdm.write(f"Saving batch {i}...")
        if save_nx:
            save_batch(graphs, i, hetero=False)

        if save_deepsnap:
            save_batch(hetero_graphs, i, hetero=True)

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
    global concept_llms, concept_ids, concept_llm_folder
    files = glob(f"../../data/text/{concept_llm_folder}/concept_embeds_*.pkl")
    for file in files:
        file_name = file.split("/")[-1].split("\\")[-1]
        with open(file, "rb") as f:
            llm_file = pickle.load(f)
            concept_llms[file_name] = llm_file

    with open("../../data/text/concept_embeds/c_ids.pkl", "rb") as f:
        concept_ids = pickle.load(f)


def get_concept_llm(concept_id):
    global concept_llms, concept_ids
    for file, ids in concept_ids.items():
        if concept_id in ids:
            return concept_llms[file].loc[concept_id]["label"]
    return None


def load_files(files_to_idx):
    global src_files, llm_files, use_llm, llm_folder
    """
    Loads files into memory
    :param files_to_idx: maps file names -> event ids
    :return:
    """

    for file_name in files_to_idx.keys():
        if file_name in src_files:
            continue
        with open(f"../../data/preprocessed_dicts/f{file_name}.pkl", "rb") as f:
            file = pickle.load(f)
            src_files[file_name] = file

        if not use_llm:
            continue

        with open(f"../../data/text/{llm_folder}/{file_name}.pkl", "rb") as f:
            file = pickle.load(f)
            llm_files[file_name] = file

    return src_files, llm_files


def add_event(graph, event_id, e_type, all_nodes, src_file, llm_file, target_ids):
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
    event_date = info["eventDate"] / 16600  # normalize by max date
    concepts = info["concepts"]
    similar = event["similar_events"]

    features = np.array([event_date], dtype=np.float32)
    target = None

    if e_type == "event":
        features = np.concatenate(
            [np.array([event_counts]), features], dtype=np.float32
        )
    else:
        target = np.array([event_counts], dtype=np.float32)

    # add llm embeddings
    if llm_file is not None:
        llm = llm_file.loc[event_id]
        title = llm["title"]
        summary = llm["summary"]
        features = np.concatenate([features, title, summary], axis=0, dtype=np.float32)

    # add node
    if e_type == "event":
        graph.add_node(event_id, node_type=e_type, node_feature=features)
    else:
        graph.add_node(
            event_id, node_type=e_type, node_feature=features, node_target=target
        )

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
        if e_from in target_ids or e_type == "event_target":
            continue

        graph.add_edge(e_from, e_to, edge_type="similar")

    # add concepts
    if use_concepts:
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
    global src_files, llm_files, concept_llms, use_llm, use_concepts

    # get files to idx
    files_to_idx, all_nodes = get_files_to_idx(target_ids, neighbor_ids, event_index)
    load_files(files_to_idx)

    graph = nx.DiGraph()
    for file_name, ids in files_to_idx.items():
        src_file = src_files[file_name]
        llm_file = llm_files[file_name] if use_llm else None

        for eid in ids:
            event_type = "event_target" if eid in target_ids else "event"
            add_event(graph, eid, event_type, all_nodes, src_file, llm_file, target_ids)

    if not use_concepts:
        return graph

    # add features to concepts
    for node in graph.nodes():
        if graph.nodes[node]["node_type"] == "concept":
            deg = graph.degree[node] - 1  # subtract self-loop
            deg = np.array([deg], dtype=np.float32)

            if not use_llm:
                graph.nodes[node]["node_feature"] = deg
                continue

            llm = get_concept_llm(node)
            graph.nodes[node]["node_feature"] = np.concatenate(
                [deg, llm], axis=0, dtype=np.float32
            )

    return graph


def main():
    subgraph_ids, event_index = load_data()
    if use_llm and use_concepts:
        load_concept_llm_files()

    subgraph_ids = subgraph_ids[:1]
    batch_generate(subgraph_ids, event_index, 1)


src_files, llm_files, concept_llms, concept_ids = {}, {}, {}, {}
concept_llm_folder = "concept_embeds"
llm_folder = "embedded"
use_llm = True
use_concepts = True

save_deepsnap = True
save_nx = False
batch_folder = "batches_llm_no_concept"

if __name__ == "__main__":
    # if batch_folder does not exist, create it
    if not os.path.exists(f"../../data/graphs/{batch_folder}"):
        os.makedirs(f"../../data/graphs/{batch_folder}")

    if not (save_deepsnap or save_nx):
        raise ValueError("Must save at least one type of graph")

    main()
