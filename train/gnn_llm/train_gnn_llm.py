import copy
import os
import torch
import deepsnap
import deepsnap.batch
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from torch.utils.data import DataLoader
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
import wandb
import optuna
import argparse

from hetero_gnn import HeteroGNN

train_args = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "hidden_size": 81,
    "epochs": 10,
    "weight_decay": 0.00002203762357664057,
    "lr": 0.003873757421883433,
    "attn_size": 48,
    "num_layers": 6,
    "aggr": "attn",
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

        meanRelative = (
            torch.sum(
                torch.abs(
                    (
                        preds["event"][non_zero_idx]
                        - graph.node_target["event"][non_zero_idx]
                    )
                    / graph.node_target["event"][non_zero_idx]
                )
            )
            / non_zero_idx.shape[0]
        )

        mse = torch.mean(torch.square(preds['event'][non_zero_idx] - graph.node_target['event'][non_zero_idx]))
        mape = torch.mean(torch.abs((preds['event'][non_zero_idx] - graph.node_target['event'][non_zero_idx]) / graph.node_target['event'][non_zero_idx]))

        tvt_scores.append((mse, L1, mape))

    # Update the best model and validation loss if the current model performs better
    if tvt_scores[1][0] < best_tvt_scores[1][0]:
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
            train_args["device"]
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
        hetero_graph.edge_index[key] = adj.t().to(train_args["device"])

    # Send node targets to device
    for key in hetero_graph.node_target:
        hetero_graph.node_target[key] = hetero_graph.node_target[key].to(
            train_args["device"]
        )


def create_split(hetero_graph):
    nEvents = hetero_graph.num_nodes("event")
    nConcepts = hetero_graph.num_nodes("concept")

    s1 = 0.7
    s2 = 0.8

    train_idx = {
        "event": torch.tensor(range(0, int(nEvents * s1))).to(train_args["device"]),
        "concept": torch.tensor(range(0, int(nConcepts * s1))).to(train_args["device"]),
    }
    val_idx = {
        "event": torch.tensor(range(int(nEvents * s1), int(nEvents * s2))).to(
            train_args["device"]
        ),
        "concept": torch.tensor(range(int(nConcepts * s1), int(nConcepts * s2))).to(
            train_args["device"]
        ),
    }
    test_idx = {
        "event": torch.tensor(range(int(nEvents * s2), nEvents)).to(
            train_args["device"]
        ),
        "concept": torch.tensor(range(int(nConcepts * s2), nConcepts)).to(
            train_args["device"]
        ),
    }

    return [train_idx, val_idx, test_idx]


def display_results(best_model):
    # TODO: Display the results of the model
    raise NotImplementedError()


def objective(trial, hetero_graph, train_idx, val_idx, test_idx):
    aggr_method = trial.suggest_categorical("aggr", ["mean", "attn"])
    attn_size = trial.suggest_int("attn_size", 16, 128) if aggr_method == "attn" else 32

    # Initialize wandb run
    wandb.init(
        project="V6_MLG_PredEvents_GNN+LMM",
        entity="mlg-events",
        dir=None,
        config={
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "hidden_size": trial.suggest_int("hidden_size", 16, 128),
            "attn_size": attn_size,  # Fixed value
            "epochs": trial.suggest_int("epochs", 150, 300),
            "num_layers": trial.suggest_int("num_layers", 1, 10),
            "aggr": aggr_method,
        },
    )

    # Use wandb config
    config = wandb.config

    # Initialize the model with the new hyperparameters
    model = HeteroGNN(
        hetero_graph,
        {
            "hidden_size": config.hidden_size,
            "attn_size": config.attn_size,
            "device": train_args["device"],
        },
        num_layers=config.num_layers,
        aggr=config.aggr,
    ).to(train_args["device"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Initialize best scores with infinity
    best_tvt_scores = (float("inf"), float("inf"), float("inf"))

    # Training loop
    for epoch in range(config.epochs):
        train_loss = train(model, optimizer, hetero_graph, train_idx)
        cur_tvt_scores, best_tvt_scores, _ = test(
            model, hetero_graph, [train_idx, val_idx, test_idx], None, best_tvt_scores
        )

        print(
            f"Epoch {epoch} Loss {train_loss:.4f} Current Train,Val,Test Scores {[score.item() for score in cur_tvt_scores]}"
        )

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_score": cur_tvt_scores[0],
                "best_val_score": best_tvt_scores[0],
            }
        )

        # Update the best validation score
        if cur_tvt_scores[1][0] < best_tvt_scores[1][0]:
            best_tvt_scores = (cur_tvt_scores[0], cur_tvt_scores[1], cur_tvt_scores[2])

    # Finish wandb runf
    wandb.finish()

    # The objective value is the best validation score
    return best_tvt_scores[1]


def hyper_parameter_tuning(hetero_graph):
    study = optuna.create_study(direction="minimize")

    train_idx, val_idx, test_idx = create_split(hetero_graph)
    study.optimize(
        lambda trial: objective(trial, hetero_graph, train_idx, val_idx, test_idx),
        n_trials=500,
    )

    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def train_model(hetero_graph, batches):

    train, val, test = dataset.split(transductive=True, split_ratio=[0.8,0.1,0.1])
    train_loader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=Batch.collate())
    val_loader = DataLoader(val, batch_size=1, shuffle=True, collate_fn=Batch.collate())
    test_loader = DataLoader(test, batch_size=1, shuffle=True, collate_fn=Batch.collate())

    best_model = None
    best_tvt_scores = (
        (float("inf"), float("inf")),
        (float("inf"), float("inf")),
        (float("inf"), float("inf")),
    )

    model = HeteroGNN(
        hetero_graph,
        train_args,
        num_layers=train_args["num_layers"],
        aggr=train_args["aggr"],
        return_embedding=False,
    ).to(train_args["device"])


    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"]
    )

    for epoch in range(train_args['epochs']):
        for train_batch, val_batch, test_batch in zip(train_loader, val_loader, test_loader):
            loss = train(model, optimizer, train_batch)

    for epoch in range(train_args["epochs"]):
        for batch in batches:
            model.hetero_graph = batch

            train_idx, val_idx, test_idx = create_split(batch)
            # Train
            #loss = train(model, optimizer, hetero_graph, train_idx)
            loss = train(model, optimizer, batch, train_idx)
            # Test for the accuracy of the model
            cur_tvt_scores, best_tvt_scores, best_model = test(
                model,
                batch,
                [train_idx, val_idx, test_idx],
                best_model,
                best_tvt_scores,
            )
            print(
                f"""Epoch: {epoch} Loss: {loss:.4f}
                Train: Mse={cur_tvt_scores[0][0].item():.4f} L1={cur_tvt_scores[0][0].item():.4f} Mape={cur_tvt_scores[0][2].item():.4f}
                Val: Mse={cur_tvt_scores[0][0].item():.4f} L1={cur_tvt_scores[1][1].item():.4f} Mape={cur_tvt_scores[0][2].item():.4f}
                Test: Mse={cur_tvt_scores[0][0].item():.4f} L1={cur_tvt_scores[2][1].item():.4f} Mape={cur_tvt_scores[0][2].item():.4f}"""
            )

    print(
        f"""Best model: {epoch} Loss: {loss:.4f}
        Train: Mse={best_tvt_scores[0][0].item():.4f} L1={best_tvt_scores[0][0].item():.4f} Mape={best_tvt_scores[0][2].item():.4f}
        Val: Mse={best_tvt_scores[0][0].item():.4f} L1={best_tvt_scores[1][1].item():.4f} Mape={best_tvt_scores[0][2].item():.4f}
        Test: Mse={best_tvt_scores[0][0].item():.4f} L1={best_tvt_scores[2][1].item():.4f} Mape={best_tvt_scores[0][2].item():.4f}"""
    )

    model = HeteroGNN(
        hetero_graph,
        train_args,
        num_layers=train_args["num_layers"],
        aggr=train_args["aggr"],
    ).to(train_args["device"])

    model.load_state_dict(torch.load("./best_model.pkl"))

    preds = model(hetero_graph.node_feature, hetero_graph.edge_index)

    cur_tvt_scores, best_tvt_scores, best_model = test(
        model, hetero_graph, [train_idx, val_idx, test_idx], best_model, best_tvt_scores
    )

    display_predictions(preds, hetero_graph, test_idx)


def display_predictions(preds, hetero_graph, test_idx):
    for i in range(test_idx["event"].shape[0]):
        if hetero_graph.node_target["event"][test_idx["event"]][i] != -1:
            print(
                i,
                preds["event"][test_idx["event"]][i],
                hetero_graph.node_target["event"][test_idx["event"]][i],
            )

def get_batches_from_pickle(folder_path):

    pickle_files = os.listdir(folder_path)

    batches = []

    for file_name in pickle_files:
        file_path = os.path.join(folder_path, file_name)
        print(file_path)
        with open(file_path, 'rb') as f:
            G = pickle.load(f)

        for node in G.nodes():
            G.nodes[node]['node_label'] = 1

        hetero_graph = HeteroGraph(G, netlib=nx, directed=True)
        batches.append(hetero_graph)

    return batches

def get_hetero_graph_dataset(folder_path):
    pickle_files = os.listdir(folder_path)

    subgraphs = []

    for file_name in pickle_files:
        file_path = os.path.join(folder_path, file_name)
        print(file_path)
        with open(file_path, 'rb') as f:
            G = pickle.load(f)

        for node in G.nodes():
            G.nodes[node]['node_label'] = 1

        hetero_graph = HeteroGraph(G, netlib=nx, directed=True)
        subgraphs.append(hetero_graph)

    dataset = GraphDataset(subgraphs, task='node')

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HeteroGNN model training or hyperparameter tuning."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "tune"],
        help="Run mode: 'train' or 'tune'",
    )
    args = parser.parse_args()

    dataset = get_hetero_graph_dataset('./subgraphs')
    train, val, test = dataset.split(transductive=True, split_ratio=[0.8,0.1,0.1])

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=Batch.collate())

    # Load the heterogeneous graph data

    with open("./1_concepts_similar_llm.pkl", "rb") as f:
        G = pickle.load(f)

    # Create a HeteroGraph object from the networkx graph

    hetero_graph = HeteroGraph(G, netlib=nx, directed=True)

    # Send all the necessary tensors to the same device
    graph_tensors_to_device(hetero_graph)

    if args.mode == "tune":
        hyper_parameter_tuning(hetero_graph)
    if args.mode == "train":
        train_model(hetero_graph, dataloader)
