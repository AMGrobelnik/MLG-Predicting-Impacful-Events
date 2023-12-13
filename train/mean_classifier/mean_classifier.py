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
    # load the networkx graph from pickle
    with open(path, 'rb') as f:
        graph = pickle.load(f)

    # extract targets and inputs from graph 
    X = []
    y = []
    for node in graph.nodes():
        if graph.nodes[node]['node_type'] == 'event_target':
            y.append(graph.nodes[node]['node_target'])
        elif graph.nodes[node]['node_type'] == 'event':
            X.append(np.array(graph.nodes[node]['node_feature'][1]))

    # convert numpy array to torch tensor
    X = np.array(X)
    y = np.array(y)
    X_torch = torch.tensor(X)
    y_torch = torch.tensor(y)

    return X, y

class MeanPredictor(nn.Module):
    def __init__(self, y):
        super(MeanPredictor, self).__init__()
        self.mean = y.mean()

    def forward(self, x):
        # Return the mean for any input
        return torch.full((x.size(0), 1), self.mean, dtype=torch.float32)

# load the train data
y_train = []
for f in os.listdir('../../data/batches/train/'):
    if f.endswith('.pkl'):
        _, y = load_data_from_graph(os.path.join('../../data/testing_batches/train', f))
        y_train.append(y)
    
y_train = torch.tensor(np.concatenate(y_train))

# load the test data
y_test = []
for f in os.listdir('../../data/batches/test/'):
    if f.endswith('.pkl'):
        _, y = load_data_from_graph(os.path.join('../../data/testing_batches/test',f))
        y_test.append(y)

y_test = torch.tensor(np.concatenate(y_test))

model = MeanPredictor(y_train)

# Predict on the test set
y_pred = model(y_test)

# Calculate MSE and MAE
mse = F.mse_loss(y_pred, y_test.float()).item()
l1 = F.l1_loss(y_pred, y_test.float()).item()

print(f"Mean Squared Error: {mse}")
print(f"L1: {l1}")

##### RESULTS ######
# Mean Squared Error: 5912.849609375
# L1: 32.43384552001953