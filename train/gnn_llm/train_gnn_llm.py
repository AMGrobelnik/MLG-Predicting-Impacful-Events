import copy
import torch
import deepsnap
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from sklearn.metrics import f1_score
from deepsnap.hetero_gnn import forward_op
from deepsnap.hetero_graph import HeteroGraph
from torch_sparse import SparseTensor, matmul

import pickle
import networkx as nx

from hetero_gnn import HeteroGNN

args = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "hidden_size": 48,
    "epochs": 500,
    "weight_decay": 0.0002930387278908051,
    "lr": 0.05091434725288385,
    "attn_size": 32,
}


def train(model, optimizer, hetero_graph, train_idx):
    """
    Trains the model on the given heterogeneous graph using the specified indices.

    :param model: The graph neural network model to train.
    :param optimizer: The optimizer used for training the model.
    :param hetero_graph: The heterogeneous graph data.
    :param train_idx: Indices for training nodes.

    :return: The training loss as a float.
    """

    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero out any existing gradients

    # Compute predictions using the model
    # TODO: Use only train_idx instead of edge_index
    # TODO: Train only on events not on concepts

    preds = model(hetero_graph.node_feature, hetero_graph.edge_index)

    # Compute the loss using model's loss function
    loss = model.loss(preds, hetero_graph.node_target, train_idx)

    loss.backward()  # Backward pass: compute gradient of the loss
    optimizer.step()  # Perform a single optimization step, updates parameters

    return loss.item()


def test(model, graph, indices, best_model, best_tvt_scores):
    """
    Tests the model on given indices and updates the best model based on validation loss.

    :param model: The trained graph neural network model.
    :param graph: The heterogeneous graph data.
    :param indices: List of indices for training, validation, and testing nodes.
    :param best_model: The current best model based on validation loss.
    :param best_val: The current best validation loss.

    :return: A tuple containing the list of losses for each dataset, the best model, and the best validation loss.
    """

    model.eval()  # Set the model to evaluation mode
    tvt_scores = []

    # Evaluate the model on each set of indices
    for index in indices:
        preds = model(graph.node_feature, graph.edge_index)

        idx = index["event"]

        # mask = y['event'][indices['event'], 0] != -1
        # non_zero_idx = torch.masked_select(indices['event'], mask)
        # preds['event'][non_zero_idx], y['event'][non_zero_idx]

        # non_zero_targets = torch.masked_select(graph.node_target['event'][indices['event']], mask)
        # non_zero_truth = torch.masked_select(graph.node_target['event'][indices['event']], mask)

        mask = graph.node_target["event"][idx, 0] != -1
        non_zero_idx = torch.masked_select(idx, mask)

        L1 = (
            torch.sum(
                torch.abs(
                    preds["event"][non_zero_idx]
                    - graph.node_target["event"][non_zero_idx]
                )
            )
            / non_zero_idx.shape[0]
        )

        tvt_scores.append(L1)

    # Update the best model and validation loss if the current model performs better
    if tvt_scores[1] < best_tvt_scores[1]:
        best_tvt_scores = tvt_scores
        # torch.to_pickle(model, 'best_model.pkl')
        # model.to_pickle('best_model.pkl')
        # best_model = copy.deepcopy(model)
        torch.save(model.state_dict(), "./best_model.pkl")

    return tvt_scores, best_tvt_scores, best_model


def graph_tensors_to_device(hetero_graph):
    for message_type in hetero_graph.message_types:
        print("TYPE", message_type)
        print("\t Feature", hetero_graph.num_node_features(message_type[0]))
        print("\t Feature", hetero_graph.num_node_features(message_type[2]))

    # Send node features to device
    for key in hetero_graph.node_feature:
        hetero_graph.node_feature[key] = hetero_graph.node_feature[key].to(
            args["device"]
        )

    # Create a torch.SparseTensor from edge_index and send it to device
    for key in hetero_graph.edge_index:
        print("KEY", key, type(key))
        print(
            "KEY NUMS",
            key,
            hetero_graph.num_nodes(key[0]),
            hetero_graph.num_nodes(key[2]),
        )

        edge_index = hetero_graph.edge_index[key]

        print(
            "MAX EDGES",
            edge_index[0].max(),
            edge_index[1].max(),
            hetero_graph.num_nodes(key[0]),
            hetero_graph.num_nodes(key[2]),
        )
        adj = SparseTensor(
            row=edge_index[0].long(),
            col=edge_index[1].long(),
            sparse_sizes=(
                hetero_graph.num_nodes(key[0]),
                hetero_graph.num_nodes(key[2]),
            ),
        )
        hetero_graph.edge_index[key] = adj.t().to(args["device"])

    # Send node targets to device
    for key in hetero_graph.node_target:
        hetero_graph.node_target[key] = hetero_graph.node_target[key].to(args["device"])


def create_split(hetero_graph):
    nEvents = hetero_graph.num_nodes("event")
    nConcepts = hetero_graph.num_nodes("concept")

    s1 = 0.7
    s2 = 0.8

    train_idx = {
        "event": torch.tensor(range(0, int(nEvents * s1))).to(args["device"]),
        "concept": torch.tensor(range(0, int(nConcepts * s1))).to(args["device"]),
    }
    val_idx = {
        "event": torch.tensor(range(int(nEvents * s1), int(nEvents * s2))).to(
            args["device"]
        ),
        "concept": torch.tensor(range(int(nConcepts * s1), int(nConcepts * s2))).to(
            args["device"]
        ),
    }
    test_idx = {
        "event": torch.tensor(range(int(nEvents * s2), nEvents)).to(args["device"]),
        "concept": torch.tensor(range(int(nConcepts * s2), nConcepts)).to(
            args["device"]
        ),
    }

    print(train_idx["event"].shape)
    print(test_idx["event"].shape)
    print(val_idx["event"].shape)

    return [train_idx, val_idx, test_idx]


def display_results():
    # TODO: Display the results of the model
    raise NotImplementedError()


if __name__ == "__main__":
    # Load the heterogeneous graph data
    with open("./1_concepts_similar_llm.pkl", "rb") as f:
        G = pickle.load(f)

    # Create a HeteroGraph object from the networkx graph
    hetero_graph = HeteroGraph(G, netlib=nx, directed=True)

    # Send all the necessary tensors to the same device
    graph_tensors_to_device(hetero_graph)

    # Create a split of the graph data
    train_idx, val_idx, test_idx = create_split(hetero_graph)

    best_model = None
    best_tvt_scores = (float("inf"), float("inf"), float("inf"))

    model = HeteroGNN(hetero_graph, args, num_layers=2, aggr="attn").to(args["device"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )

    for epoch in range(args["epochs"]):
        # Train
        loss = train(model, optimizer, hetero_graph, train_idx)
        # Test for the accuracy of the model
        cur_tvt_scores, best_tvt_scores, best_model = test(
            model,
            hetero_graph,
            [train_idx, val_idx, test_idx],
            best_model,
            best_tvt_scores,
        )
        print(
            f"Epoch {epoch} Loss {loss:.4f} Current Train,Val,Test Scores {[score.item() for score in cur_tvt_scores]}"
        )

    print("Best Train,Val,Test Scores", [score.item() for score in best_tvt_scores])