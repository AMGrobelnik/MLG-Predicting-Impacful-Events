import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import networkx as nx
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
import os

def load_data_from_graph(path):
    # Load the networkx graph from pickle
    with open(path, 'rb') as f:
        graph = pickle.load(f)

    X = []
    y = []
    neighbor_means = []

    for node in graph.nodes():
        if graph.nodes[node]['node_type'] == 'event_target':
            y.append(graph.nodes[node]['node_target'])
            neighbors = list(graph.neighbors(node))
            neighbor_values = [graph.nodes[neighbor]['node_target'] for neighbor in neighbors if 'node_target' in graph.nodes[neighbor]]
            neighbor_mean = np.mean(neighbor_values) if neighbor_values else 0
            neighbor_means.append(neighbor_mean)
    
    X = np.array(X)
    y = np.array(y)
    neighbor_means = np.array(neighbor_means)

    return X, y, neighbor_means

class NeighborhoodMeanPredictor(nn.Module):
    def __init__(self):
        super(NeighborhoodMeanPredictor, self).__init__()

    def forward(self, neighbor_means):
        # Use the provided neighbor means directly as predictions
        return torch.tensor(neighbor_means, dtype=torch.float32).unsqueeze(1)



# Load the train data
y_train, neighbor_means_train = [], []
for f in os.listdir('../../data/batches/train/'):
    if f.endswith('.pkl'):
        _, y, neighbor_means = load_data_from_graph(os.path.join('../../data/batches/train', f))
        y_train.append(y)
        neighbor_means_train.append(neighbor_means)

y_train = torch.tensor(np.concatenate(y_train))
neighbor_means_train = np.concatenate(neighbor_means_train)

# Load the test data
y_test, neighbor_means_test = [], []
for f in os.listdir('../../data/batches/test/'):
    if f.endswith('.pkl'):
        _, y, neighbor_means = load_data_from_graph(os.path.join('../../data/batches/test',f))
        y_test.append(y)
        neighbor_means_test.append(neighbor_means)

y_test = torch.tensor(np.concatenate(y_test))
neighbor_means_test = np.concatenate(neighbor_means_test)

model = NeighborhoodMeanPredictor()

# Predict on the test set using neighbor means
y_pred = model(neighbor_means_test)

# Calculate MSE and MAE
mse = F.mse_loss(y_pred, y_test.float()).item()
l1 = F.l1_loss(y_pred, y_test.float()).item()

print(f"Mean Squared Error: {mse}")
print(f"L1: {l1}")

#### RESULTS ###
# Mean Squared Error: 7134.11669921875
# L1: 35.161746978759766
