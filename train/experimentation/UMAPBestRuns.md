# GAT vs. SAGE (full 768d embeddings)

* Attention aggregation
```
Best model: Loss: 2467.6296
        Train: Mse=3774.5608 L1=18.5433 Mape=0.8842
        Val: Mse=3424.4846 L1=17.7640 Mape=0.8266
        Test: Mse=2320.0542 L1=18.8420 Mape=0.9459
```

* Graph SAGE aggregation
```
Best model: Loss: 2313.7004
        Train: Mse=4021.7510 L1=20.0032 Mape=0.8206
        Val: Mse=3695.0471 L1=18.9166 Mape=0.6977
        Test: Mse=2551.1804 L1=20.0978 Mape=0.8410
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

* UMAP dimensions: 0
```
Best model: Loss: 3211.1528
        Train: Mse=3800.7212 L1=18.4903 Mape=0.8052
        Val: Mse=3449.5144 L1=17.7830 Mape=0.7559
        Test: Mse=2346.9702 L1=18.6943 Mape=0.8421
Time elapsed 363.1913011074066
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


