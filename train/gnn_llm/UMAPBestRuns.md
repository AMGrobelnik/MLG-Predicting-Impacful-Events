# Attention vs mean

* Attention aggregation
```
Best model: Loss: 861.3662
        Train: Mse=1918.9591 L1=16.4033 Mape=1.7814
        Val: Mse=3435.9109 L1=19.2096 Mape=1.4332
        Test: Mse=2574.9324 L1=18.6548 Mape=2.1725
```

* Mean aggregation
```
Best model: Loss: 866.6550
        Train: Mse=1297.0402 L1=15.8553 Mape=2.1961
        Val: Mse=3656.9717 L1=20.7883 Mape=1.4172
        Test: Mse=2965.6980 L1=19.3168 Mape=1.9360
```



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


* UMAP dimensions: 10
```
Best model: 232 Loss: 74.6760
        Train: Mse=1207.9514 L1=13.5545 Mape=1.0253
        Val: Mse=7706.1157 L1=24.5581 Mape=1.0595
        Test: Mse=5270.7939 L1=20.0636 Mape=1.6485
```

* UMAP dimensions: 25
```
Best model: 232 Loss: 97.3956
        Train: Mse=212.2192 L1=9.3450 Mape=1.9516
        Val: Mse=6223.7837 L1=25.7239 Mape=2.0327
        Test: Mse=3474.9949 L1=18.8172 Mape=2.2061
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


