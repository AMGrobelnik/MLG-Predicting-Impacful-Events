import train_gnn
import optuna
import wandb
import torch
from gnn_llm.hetero_gnn import HeteroGNN
from torch.utils.data import Dataset
from pathlib import Path
from glob import glob
import pickle
import numpy as np


def hyper_parameter_tuning(train_set, validation_set, test_set):
    """
    Hyperparameter tuning for the GNN model
    :param train_set: set of training data
    :param validation_set: set of validation data
    :param test_set: set of testing data
    """
    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective(
            trial,
            train_set,
            validation_set,
            test_set,
        ),
        n_trials=500,
    )

    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def objective(trial, train_set, validation_set, test_set):
    aggr = trial.suggest_categorical("aggr", ["mean", "attn"])

    wandb.init(
        project="llm_full",
        entity="mlg-events",
        dir=None,
        config={
            "lr": trial.suggest_float("lr", 1e-6, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "hidden_size": trial.suggest_int("hidden_size", 16, 1024, log=True),
            "attn_size": trial.suggest_int("attn_size", 32, 1024, log=True),
            "epochs": trial.suggest_int("epochs", 1, 2),
            "num_layers": trial.suggest_int("num_layers", 3, 5),
            "aggr": aggr,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
    )

    config = wandb.config

    def epoch_log(epoch, train_losses, val_losses):
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

    _, best_loss = train_gnn.train_model(
        train_set, validation_set, config, log=epoch_log
    )
    wandb.log({"best_metric_val_loss": best_loss})

    test_losses = train_gnn.test_model(train_set, test_set, config)

    wandb.log(
        {
            "test_l1_loss": test_losses[0],
            "test_mse_loss": test_losses[1],
            "test_mape_loss": test_losses[2],
        }
    )
    wandb.finish()

    return best_loss


class BatchDataset(Dataset):
    """
    Dataset for loading heterograph objects of data from a folder.
    TODO: unload from RAM if needed
    """

    def __init__(self, folder_path, keep_in_memory=False):
        super().__init__()
        folder_path = Path(folder_path)
        batch_files = glob(str(folder_path / "batch*.pkl"))

        self.folder_path = folder_path
        self.batch_files = batch_files
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


if __name__ == "__main__":
    # Load the data
    base_dir = "../data/graphs/batches/batches_llm_full/"
    train_dataset = BatchDataset(base_dir + "train")
    val_dataset = BatchDataset(base_dir + "val")
    test_dataset = BatchDataset(base_dir + "test")

    hyper_parameter_tuning(train_dataset, val_dataset, test_dataset)
