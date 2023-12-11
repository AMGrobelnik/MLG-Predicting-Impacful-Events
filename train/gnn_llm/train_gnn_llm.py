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
import wandb
import optuna
import argparse
from multiprocessing.pool import ThreadPool

from hetero_gnn import HeteroGNN
import os
from tqdm import tqdm

train_args = {
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": "cuda",
    "hidden_size": 81,
    "epochs": 10,
    "weight_decay": 0.00002203762357664057,
    "lr": 0.003873757421883433,
    "attn_size": 32,
    "num_layers": 6,
    "aggr": "attn",
}


def train(model, optimizer, train_graph):
    """
    Trains the model on the given heterogeneous graph using the specified indices.

    :param model: The graph neural network model to train.
    :param optimizer: The optimizer used for training the model.
    :param hetero_graph: The heterogeneous graph data.
    :param train_idx: Indices for training nodes.

    :return: The training loss as a float.
    """

    # model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero out any existing gradients

    preds = model(train_graph.node_feature, train_graph.edge_index)

    # Compute the loss using model's loss function
    loss = model.loss(preds, train_graph.node_target)

    loss.backward()  # Backward pass: compute gradient of the loss
    optimizer.step()  # Perform a single optimization step, updates parameters

    return loss.item()


def test(model, graph):
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

    # TODO:
    preds = model(graph.node_feature, graph.edge_index)

    L1 = (
        torch.sum(torch.abs(preds["event_target"] - graph.node_target["event_target"]))
        / preds["event_target"].shape[0]
    )

    # print("PRED LOSS EVAL")
    # print(preds["event_target"])
    # print(preds["event_target"].shape)
    mse = torch.mean(
        torch.square(preds["event_target"] - graph.node_target["event_target"])
    )
    mape = torch.mean(
        torch.abs(
            (preds["event_target"] - graph.node_target["event_target"])
            / graph.node_target["event_target"]
        )
    )

    return (L1, mse, mape)


def offload_from_device(hetero_graph):
    # Send node features to device
    for key in hetero_graph.node_feature:
        hetero_graph.node_feature[key].to("cpu")

    for key in hetero_graph.edge_index:
        hetero_graph.edge_index[key].to("cpu")

    for key in hetero_graph.node_target:
        hetero_graph.node_target[key].to("cpu")


def graph_tensors_to_device(hetero_graph, hetero_graph_cpu):
    # G2 = nx.DiGraph()
    # G2.add_node(1)
    # G2.add_node(2)
    # G2.add_edge(1, 2)
    # hetero_graph = HeteroGraph(G=None)
    # hetero_graph.node_feature = {}
    # hetero_graph.edge_index = {}
    # hetero_graph.node_target = {}

    # for message_type in hetero_graph_cpu.message_types:
    # print("TYPE", message_type)
    # print("\t Feature", hetero_graph_cpu.num_node_features(message_type[0]))
    # print("\t Feature", hetero_graph_cpu.num_node_features(message_type[2]))

    # Send node features to device
    for key in hetero_graph_cpu.node_feature:
        hetero_graph.node_feature[key] = hetero_graph_cpu.node_feature[key].to(
            train_args["device"]
        )

    # Create a torch.SparseTensor from edge_index and send it to device
    for key in hetero_graph_cpu.edge_index:
        # print("KEY", key, type(key))
        # print(
        #     "KEY NUMS",
        #     key,
        #     hetero_graph_cpu.num_nodes(key[0]),
        #     hetero_graph_cpu.num_nodes(key[2]),
        # )

        edge_index = hetero_graph_cpu.edge_index[key]
        # print(list(edge_index))

        # print(
        #     "MAX EDGES",
        #     edge_index[0].max(),
        #     edge_index[1].max(),
        #     hetero_graph_cpu.num_nodes(key[0]),
        #     hetero_graph_cpu.num_nodes(key[2]),
        # )
        adj = SparseTensor(
            row=edge_index[0].long(),
            col=edge_index[1].long(),
            sparse_sizes=(
                hetero_graph_cpu.num_nodes(key[0]),
                hetero_graph_cpu.num_nodes(key[2]),
            ),
        )
        hetero_graph.edge_index[key] = adj.t().to(train_args["device"])

    # Send node targets to device
    for key in hetero_graph_cpu.node_target:
        hetero_graph.node_target[key] = hetero_graph_cpu.node_target[key].to(
            train_args["device"]
        )

    return hetero_graph


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


def objective(
    trial,
    train_batches,
    val_batches,
    test_batches,
    train_batches_cpu,
    val_batches_cpu,
    test_batches_cpu,
):
    # Initialize wandb run
    aggr = trial.suggest_categorical("aggr", ["mean", "attn"])
    wandb.init(
        project="V11_MLG_PredEvents_GNN+LMM",
        entity="mlg-events",
        dir=None,
        config={
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "hidden_size": trial.suggest_int("hidden_size", 16, 128),
            "attn_size": trial.suggest_int("attn_size", 16, 128)
            if aggr == "attn"
            else 32,
            "epochs": trial.suggest_int("epochs", 150, 300),
            "num_layers": trial.suggest_int("num_layers", 1, 10),
            "aggr": aggr,
        },
    )

    # Use wandb config
    config = wandb.config

    hetero_graph = train_batches[0]
    hetero_graph_cpu = train_batches_cpu[0]
    hetero_graph_gpu = graph_tensors_to_device(hetero_graph, hetero_graph_cpu)

    model = HeteroGNN(
        hetero_graph_gpu,
        config,
        num_layers=config["num_layers"],
        aggr=config["aggr"],
        return_embedding=False,
    ).to(train_args["device"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    best_loss = float("inf")
    best_epoch = -1
    for epoch in range(config["epochs"]):
        train_losses = np.zeros(4)

        for train_batch, train_batch_cpu in tqdm(
            zip(train_batches, train_batches_cpu), total=len(train_batches)
        ):
            # if train_batch != train_batches[0] and epoch == 0 or True:
            train_batch_gpu = graph_tensors_to_device(train_batch, train_batch_cpu)

            model.hetero_graph = train_batch_gpu

            loss_train = train(model, optimizer, train_batch_gpu)
            (L1_train, mse_train, mape_train) = test(model, model.hetero_graph)

            train_losses += np.array(
                [loss_train, L1_train.item(), mse_train.item(), mape_train.item()]
            )

            # offload_from_device(train_batch_gpu)

        train_losses /= len(train_batches)

        val_losses = np.zeros(3)

        for val_batch, val_batch_cpu in tqdm(
            zip(val_batches, val_batches_cpu), total=len(val_batches)
        ):
            # if epoch == 0 or True:
            val_batch_gpu = graph_tensors_to_device(val_batch, val_batch_cpu)

            model.hetero_graph = val_batch_gpu

            # test val
            (L1_val, mse_val, mape_val) = test(model, model.hetero_graph)
            val_losses += np.array([L1_val.item(), mse_val.item(), mape_val.item()])

            # offload_from_device(val_batch_gpu)

        val_losses /= len(val_batches)

        ## CHANGE THIS IF YOU WANT DIFFERENT EVALUATION METRIC
        if val_losses[0] < best_loss:
            best_epoch = epoch
            best_loss = val_losses[0]
            torch.save(model.state_dict(), "./best_model.pkl")

        print(
            f"""Epoch: {epoch} Backprop_Loss: {train_losses[0]:.1f}
            Train: Mse={train_losses[2]:.1f} L1={train_losses[1]:.1f} Mape={train_losses[3]:.1f}
            Val: Mse={val_losses[1]:.1f} L1={val_losses[0]:.1f} Mape={val_losses[2]:.1f}
            """
        )

        # Log metrics to wandb
        wandb.log(
            {
                "train_mse": train_losses[2],
                "train_l1": train_losses[1],
                "train_mape": train_losses[3],
                "val_mse": val_losses[1],
                "val_l1": val_losses[0],
                "val_mape": val_losses[2],
                # "epoch": epoch,
                "train_loss": train_losses[0],
            }
        )

    print(f"Best model: Epoch {best_epoch}, Best Metric Val Loss: {best_loss:.1f}")
    wandb.log({"best_metric_val_loss": best_loss})
    ## TEST

    model = HeteroGNN(
        hetero_graph_gpu,
        config,
        num_layers=config["num_layers"],
        aggr=config["aggr"],
    ).to(train_args["device"])

    model.load_state_dict(torch.load("./best_model.pkl"))

    test_losses = np.zeros(3)

    for test_batch, test_batch_cpu in tqdm(
        zip(test_batches, test_batches_cpu), total=len(test_batches)
    ):
        # if epoch == 0 or True:
        test_batch_gpu = graph_tensors_to_device(test_batch, test_batch_cpu)

        model.hetero_graph = test_batch_gpu

        # test
        (L1_test, mse_test, mape_test) = test(model, model.hetero_graph)
        test_losses += np.array([L1_test.item(), mse_test.item(), mape_test.item()])

        preds = model(hetero_graph.node_feature, hetero_graph.edge_index)

        display_predictions(preds, hetero_graph)

    # cur_tvt_scores, best_tvt_scores, best_model = test(
    #     model, hetero_graph, [train_idx, val_idx, test_idx], best_model, best_tvt_scores
    # )
    test_losses /= len(test_batches)

    # Print the test losses
    print(
        f"""Test Losses: 
        Mse={test_losses[1]:.1f} L1={test_losses[0]:.1f} Mape={test_losses[2]:.1f}
        """
    )

    wandb.log(
        {
            "test_l1_loss": test_losses[0],
            "test_mse_loss": test_losses[1],
            "test_mape_loss": test_losses[2],
        }
    )

    # Finish wandb run
    wandb.finish()

    # The objective value is the best validation score
    return best_loss

    # # The objective value is the best validation score
    # return best_tvt_scores[1]

    # Initialize the model with the new hyperparameters
    # model = HeteroGNN(
    #     hetero_graph,
    #     {
    #         "hidden_size": config.hidden_size,
    #         "attn_size": config.attn_size,
    #         "device": train_args["device"],
    #     },
    #     num_layers=config.num_layers,
    #     aggr=config.aggr,
    # ).to(train_args["device"])
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    # )

    # # Initialize best scores with infinity
    # best_tvt_scores = (float("inf"), float("inf"), float("inf"))

    # # Training loop
    # for epoch in range(config.epochs):
    #     train_loss = train(model, optimizer, hetero_graph, train_idx)
    #     cur_tvt_scores, best_tvt_scores, _ = test(
    #         model, hetero_graph, [train_idx, val_idx, test_idx], None, best_tvt_scores
    #     )

    #     print(
    #         f"Epoch {epoch} Loss {train_loss:.1f} Current Train,Val,Test Scores {[score.item() for score in cur_tvt_scores]}"
    #     )

    #     # Log metrics to wandb
    #     wandb.log(
    #         {
    #             "epoch": epoch,
    #             "train_loss": train_loss,
    #             "val_score": cur_tvt_scores[0],
    #             "best_val_score": best_tvt_scores[0],
    #         }
    #     )

    #     # Update the best validation score
    #     if cur_tvt_scores[1][0] < best_tvt_scores[1][0]:
    #         best_tvt_scores = (cur_tvt_scores[0], cur_tvt_scores[1], cur_tvt_scores[2])

    # # Finish wandb runf
    # wandb.finish()

    # # The objective value is the best validation score
    # return best_tvt_scores[1]


def hyper_parameter_tuning(
    train_batches,
    val_batches,
    test_batches,
    train_batches_cpu,
    val_batches_cpu,
    test_batches_cpu,
):
    study = optuna.create_study(direction="minimize")

    # train_idx, val_idx, test_idx = create_split(hetero_graph)
    study.optimize(
        lambda trial: objective(
            trial,
            train_batches,
            val_batches,
            test_batches,
            train_batches_cpu,
            val_batches_cpu,
            test_batches_cpu,
        ),
        n_trials=100,
    )

    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def train_model(
    train_batches,
    val_batches,
    test_batches,
    train_batches_cpu,
    val_batches_cpu,
    test_batches_cpu,
):
    hetero_graph = train_batches[0]
    hetero_graph_cpu = train_batches_cpu[0]
    hetero_graph_gpu = graph_tensors_to_device(hetero_graph, hetero_graph_cpu)

    model = HeteroGNN(
        hetero_graph_gpu,
        train_args,
        num_layers=train_args["num_layers"],
        aggr=train_args["aggr"],
        return_embedding=False,
        mask_unknown=True,
    ).to(train_args["device"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"]
    )

    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(train_args["epochs"]):
        train_losses = np.zeros(4)

        for train_batch, train_batch_cpu in tqdm(
            zip(train_batches, train_batches_cpu), total=len(train_batches)
        ):
            # if train_batch != train_batches[0] and epoch == 0 or True:
            train_batch_gpu = graph_tensors_to_device(train_batch, train_batch_cpu)

            model.hetero_graph = train_batch_gpu

            loss_train = train(model, optimizer, train_batch_gpu)
            (L1_train, mse_train, mape_train) = test(model, model.hetero_graph)

            train_losses += np.array(
                [loss_train, L1_train.item(), mse_train.item(), mape_train.item()]
            )

            # offload_from_device(train_batch_gpu)

        train_losses /= len(train_batches)

        val_losses = np.zeros(3)

        for val_batch, val_batch_cpu in tqdm(
            zip(val_batches, val_batches_cpu), total=len(val_batches)
        ):
            # if epoch == 0 or True:
            val_batch_gpu = graph_tensors_to_device(val_batch, val_batch_cpu)

            model.hetero_graph = val_batch_gpu

            # test val
            (L1_val, mse_val, mape_val) = test(model, model.hetero_graph)
            val_losses += np.array([L1_val.item(), mse_val.item(), mape_val.item()])

            # offload_from_device(val_batch_gpu)

        val_losses /= len(val_batches)

        ## CHANGE THIS IF YOU WANT DIFFERENT EVALUATION METRIC
        if val_losses[0] < best_loss:
            best_loss = val_losses[0]
            best_epoch = epoch
            torch.save(model.state_dict(), "./best_model.pkl")

        print(
            f"""Epoch: {epoch} Backprop_Loss: {train_losses[0]:.1f}
            Train: Mse={train_losses[2]:.1f} L1={train_losses[1]:.1f} Mape={train_losses[3]:.1f}
            Val: Mse={val_losses[1]:.1f} L1={val_losses[0]:.1f} Mape={val_losses[2]:.1f}
            """
        )

    print(f"""Best model: Epoch {best_epoch}, Best Metric Val Loss: {best_loss:.1f}""")

    ## TEST

    model = HeteroGNN(
        hetero_graph_gpu,
        train_args,
        num_layers=train_args["num_layers"],
        aggr=train_args["aggr"],
    ).to(train_args["device"])

    model.load_state_dict(torch.load("./best_model.pkl"))

    test_losses = np.zeros(3)

    for test_batch, test_batch_cpu in tqdm(
        zip(test_batches, test_batches_cpu), total=len(test_batches)
    ):
        # if epoch == 0 or True:
        test_batch_gpu = graph_tensors_to_device(test_batch, test_batch_cpu)

        model.hetero_graph = test_batch_gpu

        # test
        (L1_test, mse_test, mape_test) = test(model, model.hetero_graph)
        test_losses += np.array([L1_test.item(), mse_test.item(), mape_test.item()])

        preds = model(hetero_graph.node_feature, hetero_graph.edge_index)

        display_predictions(preds, hetero_graph)

    # cur_tvt_scores, best_tvt_scores, best_model = test(
    #     model, hetero_graph, [train_idx, val_idx, test_idx], best_model, best_tvt_scores
    # )
    test_losses /= len(test_batches)

    # Print the test losses
    print(
        f"""Test Losses: 
        Mse={test_losses[1]:.1f} L1={test_losses[0]:.1f} Mape={test_losses[2]:.1f}
        """
    )


# dsff


def display_predictions(preds, hetero_graph):
    print("Index | Predicted Value | Actual Value | Difference")
    print("-" * 70)  # Adjust the length of the separator line

    for i in range(hetero_graph.node_target["event_target"].shape[0]):
        predicted_value = round(preds["event_target"][i].item(), 2)
        actual_value = round(hetero_graph.node_target["event_target"][i].item(), 2)
        difference = round(
            abs(predicted_value - actual_value), 2
        )  # Calculate and round the absolute difference

        print(f"{i:<6} | {predicted_value:<16} | {actual_value:<13} | {difference:<10}")


def get_batches_from_pickle(folder_path):
    pickle_files = os.listdir(folder_path)

    batches = []
    cpu_batches = []

    for file_name in pickle_files:
        file_path = os.path.join(folder_path, file_name)
        print(file_path)
        with open(file_path, "rb") as f:
            G = pickle.load(f)
        # G_cpu = copy.deepcopy(G)
        # with open(file_path, "rb") as f:
        #     G_cpu = pickle.load(f)

        # print(type(G))

        hetero_graph = HeteroGraph(G, netlib=nx, directed=True)
        hetero_graph_cpu = HeteroGraph(G, netlib=nx, directed=True)

        hetero_graph["node_target"] = hetero_graph._get_node_attributes("node_target")
        hetero_graph_cpu["node_target"] = hetero_graph_cpu._get_node_attributes(
            "node_target"
        )

        # graph_tensors_to_device(hetero_graph)
        batches.append(hetero_graph)
        cpu_batches.append(hetero_graph_cpu)

    return batches, cpu_batches


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

    output_directory = "../../data/graphs/batches/"
    # batches = get_batches_from_pickle('../../data/graphs/neighborhood_sampling')
    train_batches, train_batches_cpu = get_batches_from_pickle(
        os.path.join(output_directory, "train")
    )
    test_batches, test_batches_cpu = get_batches_from_pickle(
        os.path.join(output_directory, "test")
    )
    val_batches, val_batches_cpu = get_batches_from_pickle(
        os.path.join(output_directory, "val")
    )

    # Load the heterogeneous graph data

    # with open("./1_concepts_similar_llm.pkl", "rb") as f:
    # G = pickle.load(f)

    # Create a HeteroGraph object from the networkx graph
    # hetero_graph = HeteroGraph(G, netlib=nx, directed=True)

    # print("DIM: ", hetero_graph.node_feature["event"][0].shape)

    # hetero_graph = HeteroGraph(G, netlib=nx, directed=True)
    # hetero_graph = None
    # Send all the necessary tensors to the same device
    # graph_tensors_to_device(hetero_graph)
    if args.mode == "tune":
        hyper_parameter_tuning(
            train_batches,
            val_batches,
            test_batches,
            train_batches_cpu,
            val_batches_cpu,
            test_batches_cpu,
        )
    if args.mode == "train":
        train_model(
            train_batches,
            val_batches,
            test_batches,
            train_batches_cpu,
            val_batches_cpu,
            test_batches_cpu,
        )
