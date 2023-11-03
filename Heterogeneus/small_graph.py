import torch

node_features = {
    "event": torch.tensor([
                [0, 0, 0],   # event 0
                [1, 1, 1]    # event 1
    ]),
    "concept": torch.tensor([[2, 2, 2]])
}

edge_index = {
    ("event", "similar", "event"): torch.tensor([[0,1],[1,0]]),
    ("event", "related", "concept"): torch.tensor([[0,0],[0,0]]),
    ("concept", "related", "event"): torch.tensor([[0,0],[0,0]])
}