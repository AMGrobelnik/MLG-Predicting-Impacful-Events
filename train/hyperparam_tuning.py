import train_gnn
import optuna
import wandb
from train_gnn import BatchDataset


def hyper_parameter_tuning(
    train_set, validation_set, test_set, project_name: str, get_config
):
    """
    Hyperparameter tuning for the GNN model
    :param train_set: set of training data
    :param validation_set: set of validation data
    :param test_set: set of testing data
    """
    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda trial: objective(
            trial, train_set, validation_set, test_set, project_name, get_config(trial)
        ),
        n_trials=500,
    )

    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def objective(trial, train_set, validation_set, test_set, project, config):
    wandb.init(
        project=project,
        entity="mlg-events",
        dir=None,
        config=config,
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


if __name__ == "__main__":
    # Load the data
    base_dir = "../data/graphs/batches/batches_llm_full/"
    train_dataset = BatchDataset(base_dir + "train")
    val_dataset = BatchDataset(base_dir + "val")
    test_dataset = BatchDataset(base_dir + "test")

    def get_config(trial):
        aggr = trial.suggest_categorical("aggr", ["mean", "attn"])

        if aggr == "mean":
            attn_size = 0
            hidden_size = trial.suggest_int("hidden_size", 64, 512, log=True)
            num_layers = trial.suggest_int("num_layers", 3, 5)
        else:
            hidden_size = trial.suggest_int("hidden_size", 16, 256, log=True)
            attn_size = trial.suggest_int("attn_size", 32, 256, log=True)
            num_layers = trial.suggest_int("num_layers", 3, 5)

        return {
                "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 1e-5, 1e-3, log=True
                ),
                "hidden_size": hidden_size,
                "attn_size": attn_size,
                "epochs": trial.suggest_int("epochs", 20, 40),
                "num_layers": num_layers,
                "aggr": aggr,
                "device": "cuda",
            }

    project_name = "test"

    hyper_parameter_tuning(
        train_dataset, val_dataset, test_dataset, project_name, get_config
    )
