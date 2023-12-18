import torch
from hetero_gnn import HeteroGNN
from torch.utils.data import Dataset
from pathlib import Path
from glob import glob
import pickle
import numpy as np


class BatchDataset(Dataset):
    """
    Dataset for loading heterograph objects of data from a folder.
    """

    def __init__(self, folder_path, keep_in_memory=False):
        super().__init__()
        folder_path = Path(folder_path)
        batch_files = glob(str(folder_path / "batch*.pkl"))

        self.folder_path = folder_path
        self.batch_files = sorted(batch_files)
        self.loaded = {}
        self.keep_in_memory = keep_in_memory

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        if self.keep_in_memory and idx in self.loaded:
            return self.loaded[idx]

        batch_file = self.batch_files[idx]

        with open(batch_file, "rb") as f:
            batch = pickle.load(f)

        if self.keep_in_memory:
            self.loaded[idx] = batch

        return batch


def train_step(model, optimizer, train_graph):
    """
    Trains the model on the given heterogeneous graph using the specified indices.
    :param model: The graph neural network model.
    :param optimizer: The optimizer for the model.
    :param train_graph: The heterogeneous graph data.
    :return: The training loss as a float.
    """

    # Make sure the model is in train mode.
    if not model.training:
        model.train()

    # Set the model's graph to the training graph
    model.hetero_graph = train_graph

    # Zero out any existing gradients
    optimizer.zero_grad()

    # get model predictions
    preds = model(train_graph.node_feature, train_graph.edge_index)

    # Compute the loss using model's loss function
    loss = model.loss(preds, train_graph.node_target)

    # Backward pass: compute gradient of the loss
    loss.backward()
    # Perform a single optimization step, updates parameters
    optimizer.step()

    # Return the loss
    return loss.item()


def evaluate(model, graph, round_preds=False):
    """
    Tests the model on given indices and updates the best model based on validation loss.

    :param model: The trained graph neural network model.
    :param graph: The heterogeneous graph data.
    :param round_preds: Whether to round the predictions to the nearest integer.
    :return: The average L1, MSE, and MAPE loss
    """
    # Make sure the model is in eval mode.
    if model.training:
        model.eval()

    # Set the model's graph to the training graph
    model.hetero_graph = graph

    # Get the model predictions
    preds = model(graph.node_feature, graph.edge_index)

    preds = preds["event_target"]
    if round_preds:
        preds = torch.round(preds)

    actual = graph.node_target["event_target"]

    # average L1
    l1 = torch.mean(torch.abs(preds - actual))

    # Mean Squared Error
    mse = torch.mean((preds - actual) ** 2)

    # Mean Absolute Percentage Error
    mape = torch.mean(torch.abs(preds - actual) / actual)

    # Move the calculated metrics to cpu
    l1 = l1.detach().cpu()
    mse = mse.detach().cpu()
    mape = mape.detach().cpu()

    return l1, mse, mape


def graph_to_device(graph, device):
    """
    Moves the graph tensors to the specified device.
    """

    def to_device(attr):
        for key in graph[attr]:
            graph[attr][key] = graph[attr][key].to(device)

    to_device("node_feature")
    to_device("node_target")
    to_device("edge_index")

    return graph


def graph_unload(graph):
    """
    Unloads the graph tensors from the device to the CPU.
    """

    def unload(attr):
        for key in graph[attr]:
            graph[attr][key] = graph[attr][key].detach().cpu()

    unload("node_feature")
    unload("node_target")
    unload("edge_index")

    return graph


def train_model(train_set: Dataset, val_set: Dataset, train_args, log=None):
    """
    Trains and evaluates the model on the given data loaders.
    :param train_set: set of training data
    :param val_set: set of validation data
    :param train_args: dictionary of training arguments
    :param log: function to log training progress after each epoch
    :return: The trained model.
    """

    # Initialize the model with an example graph
    model = HeteroGNN(
        train_set[0],
        *train_args,
    )
    # Move the model to the GPU
    model = model.to(train_args["device"])

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"]
    )

    best_loss, best_epoch = float("inf"), -1
    for epoch in range(train_args["epochs"]):
        train_losses = []

        # Iterate over batches
        for batch in train_set:
            # Move the batch to the GPU if available
            batch = graph_to_device(batch, train_args["device"])

            # Perform one training step
            loss = train_step(model, optimizer, batch)

            # Evaluate
            l1, mse, mape = evaluate(model, batch)
            train_losses.append([loss, l1, mse, mape])

            # Unload the batch from the GPU
            graph_unload(batch)

        # Get train loss
        train_losses = np.array(train_losses)
        train_losses = np.mean(train_losses, axis=0)

        # Evaluate on validation set
        val_losses = []
        for batch in val_set:
            # Move the batch to the GPU if available
            batch = graph_to_device(batch, train_args["device"])

            # Evaluate
            l1, mse, mape = evaluate(model, batch)
            val_losses.append([l1, mse, mape])

            # Unload the batch from the GPU
            graph_unload(batch)

        # Get validation loss
        val_losses = np.array(val_losses)
        val_losses = np.mean(val_losses, axis=0)

        # Determine if this is the best model, based one L1 loss
        if val_losses[0] < best_loss:
            best_loss = val_losses[0]
            best_epoch = epoch

            # save the model
            torch.save(model.state_dict(), "./best_model.pkl")

        print(
            f"""Epoch: {epoch} Backprop_Loss: {train_losses[0]:.1f}
            Train: Mse={train_losses[2]:.1f} L1={train_losses[1]:.1f} Mape={train_losses[3]:.1f}
            Val: Mse={val_losses[1]:.1f} L1={val_losses[0]:.1f} Mape={val_losses[2]:.1f}
            """
        )

        torch.cuda.empty_cache()
        if log:
            log(epoch, train_losses, val_losses)

    # unload the model from the GPU
    del model
    torch.cuda.empty_cache()

    print(f"""Best model: Epoch {best_epoch}, Best Metric Val Loss: {best_loss:.1f}""")
    return best_epoch, best_loss


def test_model(train_set, test_set, train_args):
    """
    Tests the best model
    :param test_set: set of testing data
    """

    model = HeteroGNN(
        train_set[0],
        *train_args,
    )
    model = model.to(train_args["device"])

    # load the best model's parameters
    model.load_state_dict(torch.load("./best_model.pkl"))

    test_losses = []
    for batch in test_set:
        # Move the batch to the GPU if available
        batch = graph_to_device(batch, train_args["device"])

        # Evaluate
        l1, mse, mape = evaluate(model, batch, round_preds=True)
        test_losses.append([l1, mse, mape])

        predictions = model(batch.node_feature, batch.edge_index)
        display_predictions(predictions, batch, round_preds=True)

        # Unload the batch from the GPU
        graph_unload(batch)

    # Unload the model from the GPU
    del model
    torch.cuda.empty_cache()

    # Get test loss
    test_losses = np.array(test_losses)
    test_losses = np.mean(test_losses, axis=0)

    print(
        f"""Test Losses: 
        Mse={test_losses[1]:.1f} L1={test_losses[0]:.1f} Mape={test_losses[2]:.1f}
        """
    )

    return test_losses


def display_predictions(preds, hetero_graph, round_preds=False):
    print("Index | Predicted Value | Actual Value | Difference")
    print("-" * 70)  # Adjust the length of the separator line

    for i in range(hetero_graph.node_target["event_target"].shape[0]):
        predicted_value = round(preds["event_target"][i].item(), 2)

        if round_preds:
            predicted_value = int(predicted_value)

        actual_value = round(hetero_graph.node_target["event_target"][i].item(), 2)
        difference = round(
            abs(predicted_value - actual_value), 2
        )  # Calculate and round the absolute difference

        print(f"{i:<6} | {predicted_value:<16} | {actual_value:<13} | {difference:<10}")


if __name__ == "__main__":
    train_args = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "hidden_size": 81,
        "epochs": 30,
        "weight_decay": 0.00002203762357664057,
        "lr": 0.003873757421883433,
        "attn_size": 32,
        "num_layers": 6,
        "aggr": "attn",
    }

    # Load the data
    base_dir = "../data/graphs/batches/gnn_only/"
    train_dataset = BatchDataset(base_dir + "train")
    val_dataset = BatchDataset(base_dir + "val")
    test_dataset = BatchDataset(base_dir + "test")

    # Train the model
    train_model(train_dataset, val_dataset, train_args)

    # Test the model
    test_model(train_dataset, test_dataset, train_args)
