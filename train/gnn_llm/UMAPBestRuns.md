## UMAP 100 dim

Hyperparams:
```
train_args = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "hidden_size": 81,
    "epochs": 233,
    "weight_decay": 0.00002203762357664057,
    "lr": 0.003873757421883433,
    "attn_size": 32,
    "num_layers": 6,
    "aggr": "attn",
}
```


* UMAP dimensions: 50
```
Best model: 232 Loss: 74.6760
        Train: Mse=1207.9514 L1=13.5545 Mape=1.0253
        Val: Mse=7706.1157 L1=24.5581 Mape=1.0595
        Test: Mse=5270.7939 L1=20.0636 Mape=1.6485
```


* UMAP dimensions: 100
```
Best model: 232 Loss: 78.9376
        Train: Mse=235.3967 L1=9.2063 Mape=1.7515
        Val: Mse=6858.7734 L1=28.2024 Mape=2.7877
        Test: Mse=4068.9812 L1=23.1028 Mape=2.9926
```


