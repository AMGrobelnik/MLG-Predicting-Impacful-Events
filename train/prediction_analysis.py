from train_gnn import graph_to_device, graph_unload
from hetero_gnn import HeteroGNN
import torch
import numpy as np


def get_model(train_set, args, return_embeddings, model_params=None):
    """
    Builds the model
    :param train_set: set of training data. It is used to initialize the model's matrices in the same order as was during training.
    :param return_embeddings: whether to return the embeddings or predictions
    :param args: dictionary of training arguments
    :param model_params: path to saved model parameters
    :return:
    """
    model = HeteroGNN(
        train_set[0],
        args,
        num_layers=args["num_layers"],
        aggr=args["aggr"],
        return_embedding=return_embeddings,
    )
    model = model.to(args["device"])

    # load the best model's parameters
    if model_params:
        model.load_state_dict(torch.load(model_params))

    return model


def get_predictions(train_set, test_set, train_args):
    """
    Gets the predictions for the best model
    :param train_set: set of training batches
    :param test_set: set of testing batches
    :param train_args: dictionary of training arguments
    :return: list of predictions, one for each batch
    """
    model = get_model(
        train_set, train_args, return_embeddings=False, model_params="./best_model.pkl"
    )

    predictions = []
    for batch in test_set:
        # Move the batch to the GPU if available
        batch = graph_to_device(batch, train_args["device"])

        # Evaluate
        batch_predictions = model(batch.node_feature, batch.edge_index)
        batch_predictions = batch_predictions["event_target"].detach().cpu().numpy()
        predictions.append(batch_predictions)

        # Unload the batch from the GPU
        graph_unload(batch)

    # Unload the model from the GPU
    del model
    torch.cuda.empty_cache()

    return predictions


def get_embeddings(train_set, test_set, train_args):
    """
    Gets the embeddings for the best model
    :param train_set: set of training batches
    :param test_set: set of testing batches
    :param train_args: dictionary of training arguments
    """
    model = get_model(
        train_set, train_args, return_embeddings=True, model_params="./best_model.pkl"
    )

    embeddings = []
    for batch in test_set:
        # Move the batch to the GPU if available
        batch = graph_to_device(batch, train_args["device"])

        # Evaluate
        batch_embeddings = model(batch.node_feature, batch.edge_index)
        for key, value in batch_embeddings.items():
            batch_embeddings[key] = value.detach().cpu().numpy()
        embeddings.append(batch_embeddings)

        # Unload the batch from the GPU
        graph_unload(batch)

    # Unload the model from the GPU
    del model
    torch.cuda.empty_cache()

    return embeddings


def check_graph_matching(hetero_graphs, nx_graphs):
    """
    Checks if the hetero graphs and their corresponding networkx graphs match
    by comparing the node targets
    """
    if len(hetero_graphs) != len(nx_graphs):
        raise ValueError("The number of heterographs and networkx graphs must match")

    for hetero_graph, nx_graph in zip(hetero_graphs, nx_graphs):
        targets = hetero_graph.node_target["event_target"]

        for _, node_id, node_type in iterate_nodes(hetero_graph, nx_graph):
            assert node_type[0] == node_id[0]

        for i, node_id, node_type in iterate_nodes(hetero_graph, nx_graph, ["event_target"]):
            data = nx_graph.nodes[node_id]
            target = data["node_target"][0]
            hg_target = targets[i].item()

            assert target == hg_target


def hetero_idx_to_ids(hetero_graph, nx_graph):
    """
    Converts the indices of the heterograph to the ids of the original networkx graph
    :param hetero_graph: the heterograph
    :param nx_graph: the original networkx graph
    :return: a dictionary mapping the heterograph indices to the original graph ids
    """
    nodes = list(nx_graph.nodes)
    hetero_ids = hetero_graph.node_to_graph_mapping

    idx_to_id = {}

    for key, indices in hetero_ids.items():
        for idx in indices:
            idx = idx.item()
            idx_to_id[idx] = nodes[idx]

    return idx_to_id


def iterate_nodes(hetero_graph, nx_graph, node_types=None):
    if node_types is None:
        node_types = hetero_graph.node_types

    idx_to_ids = hetero_idx_to_ids(hetero_graph, nx_graph)
    for node_type in node_types:
        indices = hetero_graph.node_to_graph_mapping[node_type]
        for i, idx in enumerate(indices):
            idx = idx.item()
            node_id = idx_to_ids[idx]
            yield i, node_id, node_type


def get_results(hetero_graphs, nx_graphs, train_set, train_args):
    """
    Adds the predictions and gnn embeddings to the original networkx graph
    :param hetero_graphs: batched heterographs
    :param nx_graphs: list of corresponding networkx graphs
    :param train_set: set of training batches
    :param train_args: dictionary of training arguments
    :return:
    """

    check_graph_matching(hetero_graphs, nx_graphs)

    # Get the predictions and embeddings
    predictions = get_predictions(train_set, hetero_graphs, train_args)
    embeddings = get_embeddings(train_set, hetero_graphs, train_args)

    for hetero, preds, embeds, nx_graph in zip(
        hetero_graphs, predictions, embeddings, nx_graphs
    ):
        # iterate over all node types
        for i, node_id, node_type in iterate_nodes(hetero, nx_graph):
            # add embeddings to all nodes
            nx_graph.nodes[node_id]["embedding"] = embeds[node_type][i]
            # add predictions to target nodes
            if node_type == "event_target":
                nx_graph.nodes[node_id]["prediction"] = preds[i]

    return nx_graphs
