import networkx as nx
import torch
from glob import glob
import pickle
from tqdm import tqdm


def add_dummy_feats_to_concepts(graph: nx.DiGraph):
    """
    Adds dummy features to concepts
    :param graph: graph
    :return:
    """
    for node in graph.nodes(data=True):
        node_id, node_data = node
        node_type = node_data["node_type"]
        if node_type != "concept":
            continue

        graph.nodes[node_id]["node_feature"] = torch.zeros(1)


def load_graph(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def save_graph(graph, file):
    with open(file, "wb") as f:
        pickle.dump(graph, f)


def main():
    batch_files = sorted(glob("../../data/graphs/batches/batch-*.pkl"))
    for batch_file in tqdm(batch_files, ncols=100, desc="Processing"):
        graph = load_graph(batch_file)
        add_dummy_feats_to_concepts(graph)
        save_graph(graph, batch_file)
