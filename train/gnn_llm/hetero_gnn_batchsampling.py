import copy
import torch
import deepsnap
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from sklearn.metrics import f1_score
from deepsnap.hetero_gnn import forward_op
from deepsnap.hetero_graph import HeteroGraph
from torch_sparse import SparseTensor, matmul
from torchmetrics.regression import MeanAbsolutePercentageError

import random
import bisect

import pickle
import networkx as nx

train_args = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "hidden_size": 64,
    "epochs": 200,
    "weight_decay": 0.0002930387278908051,
    "lr": 0.05091434725288385,
    "attn_size": 64,
    "num_layers": 4,
    "aggr": "attn",
}

class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        self.lin_dst = None
        self.lin_src = None

        self.lin_update = None

        self.lin_dst = nn.Linear(in_channels_dst, out_channels)
        self.lin_src = nn.Linear(in_channels_src, out_channels)
        self.lin_update = nn.Linear(2 * out_channels, out_channels)

    def forward(
        self,
        node_feature_src,
        node_feature_dst,
        edge_index,
        size=None,
        res_n_id=None,
    ):
        return self.propagate(
            edge_index,
            node_feature_src=node_feature_src,
            node_feature_dst=node_feature_dst,
            size=size,
            res_n_id=res_n_id,
        )

    def message_and_aggregate(self, edge_index, node_feature_src):
        out = matmul(edge_index, node_feature_src, reduce="mean")

        return out

    def update(self, aggr_out, node_feature_dst, res_n_id):
        dst_out = self.lin_dst(node_feature_dst)
        aggr_out = self.lin_src(aggr_out)
        aggr_out = torch.cat([dst_out, aggr_out], -1)
        aggr_out = self.lin_update(aggr_out)

        return aggr_out


class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    def __init__(self, convs, args, aggr="mean"):
        """
        Initializes the HeteroGNNWrapperConv instance.

        :param convs: Dictionary of convolution layers for each message type.
        :param args: Arguments dictionary containing hyperparameters like hidden_size and attn_size.
        :param aggr: Aggregation method, defaults to 'mean'.
        """

        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = aggr

        # Map the index and message type
        self.mapping = {}

        # A numpy array that stores the final attention probability
        self.alpha = None

        self.attn_proj = None

        if self.aggr == "attn":
            self.attn_proj = nn.Sequential(
                nn.Linear(args["hidden_size"], args["attn_size"]),
                nn.Tanh(),
                nn.Linear(args["attn_size"], 1, bias=False),
            )

    def reset_parameters(self):
        super(HeteroGNNWrapperConv, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                layer.reset_parameters()

    def forward(self, node_features, edge_indices):
        """
        Forward pass of the model.

        :param node_features: Dictionary of node features for each node type.
        :param edge_indices: Dictionary of edge indices for each message type.
        :return: Aggregated node embeddings for each node type.
        """

        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            message_type_emb[message_key] = self.convs[message_key](
                node_feature_src,
                node_feature_dst,
                edge_index,
            )

        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}

        for (src, edge_type, dst), item in message_type_emb.items():
            mapping[len(node_emb[dst])] = (src, edge_type, dst)
            node_emb[dst].append(item)
        self.mapping = mapping

        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)

        return node_emb

    def aggregate(self, xs):
        """
        Aggregates node embeddings using the specified aggregation method.

        :param xs: List of node embeddings to aggregate.
        :return: Aggregated node embeddings as a torch.Tensor.
        """

        if self.aggr == "mean":
            xs = torch.stack(xs)
            out = torch.mean(xs, dim=0)
            return out

        elif self.aggr == "attn":
            xs = torch.stack(xs, dim=0) 
            s = self.attn_proj(xs).squeeze(-1) # Pass the xs through the attention layer
            s = torch.mean(s, dim=-1) # Average the attention scores across the source nodes
            self.alpha = torch.softmax(s, dim=0).detach() # Compute the attention probability
            out = self.alpha.reshape(-1, 1, 1) * xs
            out = torch.sum(out, dim=0)
            return out


def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
    """
    Generates convolutional layers for each message type in a heterogeneous graph.

    :param hetero_graph: The heterogeneous graph for which convolutions are to be created.
    :param conv: The convolutional layer class or constructor.
    :param hidden_size: The number of features in the hidden layer.
    :param first_layer: Boolean indicating if this is the first layer in the network.

    :return: A dictionary of convolutional layers, keyed by message type.
    """

    convs = {}

    # Extracting all types of messages/edges in the heterogeneous graph.
    all_messages_types = hetero_graph.message_types
    for message_type in all_messages_types:
        # Determine the input feature size for source and destination nodes.
        # If it's the first layer, use the feature size of the nodes.
        # Otherwise, use the hidden size, since from there on the size of embeddings
        # is the same for all nodes.
        if first_layer:
            in_channels_src = hetero_graph.num_node_features(message_type[0])
            in_channels_dst = hetero_graph.num_node_features(message_type[2])
        else:
            in_channels_src = hidden_size
            in_channels_dst = hidden_size
        out_channels = hidden_size

        # Create a convolutional layer for this message type and add it to the dictionary.
        convs[message_type] = conv(in_channels_src, in_channels_dst, out_channels)

    return convs


class HeteroGNN(torch.nn.Module):
    # def __init__(self, hetero_graph, args, num_layers, aggr="mean"):
    #     super(HeteroGNN, self).__init__()

    #     self.aggr = aggr
    #     self.hidden_size = args["hidden_size"]

    #     self.bns1 = nn.ModuleDict()
    #     self.bns2 = nn.ModuleDict()
    #     self.relus1 = nn.ModuleDict()
    #     self.relus2 = nn.ModuleDict()
    #     self.post_mps = nn.ModuleDict()
    #     self.fc = nn.ModuleDict()

    #     # Initialize the graph convolutional layers
    #     self.convs1 = HeteroGNNWrapperConv(
    #         generate_convs(
    #             hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=True
    #         ),
    #         args,
    #         self.aggr,
    #     )
    #     self.convs2 = HeteroGNNWrapperConv(
    #         generate_convs(
    #             hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=False
    #         ),
    #         args,
    #         self.aggr,
    #     )

    #     # Initialize batch normalization, ReLU, and fully connected layers for each node type
    #     all_node_types = hetero_graph.node_types
    #     for node_type in all_node_types:
    #         self.bns1[node_type] = nn.BatchNorm1d(self.hidden_size, eps=1.0)
    #         self.bns2[node_type] = nn.BatchNorm1d(self.hidden_size, eps=1.0)

    #         self.relus1[node_type] = nn.LeakyReLU()
    #         self.relus2[node_type] = nn.LeakyReLU()
    #         self.fc[node_type] = nn.Linear(self.hidden_size, 1)

    def __init__(self, hetero_graph, args, num_layers, aggr="mean"):
        super(HeteroGNN, self).__init__()
        
        # Store full graph for minbatch sampling
        self.hetero_graph = hetero_graph
        
        self.aggr = aggr
        self.device = args["device"]
        self.hidden_size = args["hidden_size"]
        self.num_layers = num_layers

        # Use a single ModuleDict for batch normalization and ReLU layers
        self.bns = nn.ModuleDict()
        self.relus = nn.ModuleDict()
        self.convs = nn.ModuleList()
        self.fc = nn.ModuleDict()

        # Initialize the first graph convolutional layer
        self.convs.append(
            HeteroGNNWrapperConv(
                generate_convs(
                    hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=True
                ),
                args,
                self.aggr,
            )
        )

        # Initialize the rest of the graph convolutional layers
        for _ in range(1, self.num_layers):
            self.convs.append(
                HeteroGNNWrapperConv(
                    generate_convs(
                        hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=False
                    ),
                    args,
                    self.aggr,
                )
            )

        # Initialize batch normalization and ReLU layers for each layer and node type
        all_node_types = hetero_graph.node_types
        for i in range(self.num_layers):
            for node_type in all_node_types:
                key_bn = f'bn_{i}_{node_type}'
                key_relu = f'relu_{i}_{node_type}'
                self.bns[key_bn] = nn.BatchNorm1d(self.hidden_size, eps=1.0)
                self.relus[key_relu] = nn.LeakyReLU()

        # Initialize fully connected layers for each node type
        for node_type in all_node_types:
            self.fc[node_type] = nn.Linear(self.hidden_size, 1)

    # def forward(self, node_feature, edge_index):
    #     """
    #     Forward pass of the model.

    #     :param node_feature: Dictionary of node features for each node type.
    #     :param edge_index: Dictionary of edge indices for each message type.
    #     :return: The output embeddings for each node type after passing through the model.
    #     """
    #     x = node_feature

    #     # Apply graph convolutional, batch normalization, and ReLU layers
    #     x = self.convs1(x, edge_index)
        #  x = forward_op(x, self.bns1)
    #     x = forward_op(x, self.relus1)

    #     x = self.convs2(x, edge_index)
    #     x = forward_op(x, self.bns2)
    #     x = forward_op(x, self.relus2)

    #     x = forward_op(x, self.fc)
    #     return x
    

    def binary_search_index(self, src_indices, node):
        """
        Performs binary search to find an index of 'node' in the sorted 'src_indices' tensor.

        :param src_indices: Source indices tensor of the edges (sorted).
        :param node: The node to search for.
        :return: An index where 'node' is found, or -1 if not found.
        """
        left = 0
        right = src_indices.size(0) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if src_indices[mid] < node:
                left = mid + 1
            elif src_indices[mid] > node:
                right = mid - 1
            else:
                return mid  # Node found
        return -1  # Node not found

    def get_neighbors(self, src_indices, dst_indices, node_idx):
        """
        Finds the immediate neighbors of a given node by looking at the edge indices.

        :param src_indices: Source indices tensor of the edges.
        :param dst_indices: Destination indices tensor of the edges.
        :param node_idx: The index of the node in the edge index tensor.
        :return: A set containing the immediate neighbor node indices.
        """
        # Find the range in the edge index where the node appears as a source
        start_idx = node_idx
        while start_idx > 0 and src_indices[start_idx - 1] == src_indices[node_idx]:
            start_idx -= 1
        end_idx = node_idx + 1
        while end_idx < src_indices.size(0) and src_indices[end_idx] == src_indices[node_idx]:
            end_idx += 1

        # Get all neighbors where the node is the source
        neighbors = set(dst_indices[start_idx:end_idx].tolist())
        return neighbors

    def get_k_hop_neighbors(self, src_indices, dst_indices, node, node_idx, k):
        """
        Recursively finds the k-hop neighborhood of a given node.

        :param src_indices: Source indices tensor of the edges.
        :param dst_indices: Destination indices tensor of the edges.
        :param node: The node from which to build the neighborhood.
        :param node_idx: The index of the node in the edge index tensor.
        :param k: The number of hops to consider.
        :return: A set containing the k-hop neighborhood node indices.
        """
        if k == 0:
            return set([node])

        # Get immediate neighbors for the current hop
        neighbors = self.get_neighbors(src_indices, dst_indices, node_idx)
        k_hop_neighbors = set(neighbors)

        # Recursively find neighbors for the next hop
        for neighbor in neighbors:
            neighbor_idx = self.binary_search_index(src_indices, neighbor)
            if neighbor_idx != -1:  # Ensure the neighbor is found
                k_hop_neighbors.update(self.get_k_hop_neighbors(src_indices, dst_indices, neighbor, neighbor_idx, k - 1))

        return k_hop_neighbors

    def gen_minibatch(self, batch_size):
        """
        Generates a minibatch by randomly sampling a starting node and getting its k-hop neighborhood.

        :param batch_size: The number of nodes to sample for the "event" node type.
        :return: A tuple containing dictionaries of sampled node features and edge indices.
        """
        node_feature_mini = {}
        edge_index_mini = {}

        # Find the message type where "event" nodes are the source
        for message_type in self.hetero_graph.message_types:
            src_type, _, dst_type = message_type
            if src_type == 'event':
                # Convert SparseTensor to COO format and get the indices
                edge_index = self.hetero_graph.edge_index[message_type].coo()
                src_indices, dst_indices = edge_index[0], edge_index[1]

                # Randomly sample a node index from the source indices tensor
                sampled_event_node_idx = torch.randint(0, src_indices.size(0), (1,), device=self.device).item()
                sampled_event_node = src_indices[sampled_event_node_idx].item()

                # Get the k-hop neighborhood of the sampled node
                k_hop_neighbors = self.get_k_hop_neighbors(src_indices, dst_indices, sampled_event_node, self.num_layers)

                # Get the node features for the sampled node and its k-hop neighborhood
                node_feature_mini['event'] = self.hetero_graph.node_feature['event'][list(k_hop_neighbors)]

                # Find all edges within the k-hop neighborhood
                # This will be done outside of this function, as we need to collect edges from all neighbors

                break  # Since we are only sampling one node, we can break after finding it

        # Collect all edges within the k-hop neighborhood
        for message_type in self.hetero_graph.message_types:
            src_type, _, dst_type = message_type
            if src_type == 'event':
                edge_index = self.hetero_graph.edge_index[message_type].coo()
                src_indices, dst_indices = edge_index[0], edge_index[1]

                # Create a mask for edges where both src and dst are in the k-hop neighborhood
                mask = torch.zeros_like(src_indices, dtype=torch.bool)
                for node in k_hop_neighbors:
                    node_mask = src_indices == node
                    mask |= node_mask

                # Apply the mask to get the subgraph edge indices
                edge_index_mini[message_type] = torch.stack([src_indices[mask], dst_indices[mask]], dim=0)

        return node_feature_mini, edge_index_mini
    
    
    def forward(self, node_feature, edge_index):
        """
        Forward pass of the model.

        :param node_feature: Dictionary of node features for each node type.
        :param edge_index: Dictionary of edge indices for each message type.
        :return: The output embeddings for each node type after passing through the model.
        """
        
        node_feature_mini, edge_index_mini = self.gen_minibatch(10)
        
        x = node_feature

        # Apply graph convolutional, batch normalization, and ReLU layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)  # Apply the i-th graph convolutional layer
            for node_type in x:
                key_bn = f'bn_{i}_{node_type}'
                key_relu = f'relu_{i}_{node_type}'
                x[node_type] = self.bns[key_bn](x[node_type])  # Apply batch normalization
                x[node_type] = self.relus[key_relu](x[node_type])  # Apply ReLU

        # Apply the final fully connected layers
        for node_type in x:
            x[node_type] = self.fc[node_type](x[node_type])

        return x
    

    def loss(self, preds, y, indices):
        """
        Computes the loss for the model.

        :param preds: Predictions made by the model.
        :param y: Ground truth target values.
        :param indices: Indices of nodes for which loss should be calculated.

        :return: The computed loss value.
        """

        mape = MeanAbsolutePercentageError().to(train_args["device"])

        loss = 0
        loss_func = torch.nn.MSELoss()
        loss_func = mape

        mask = y["event"][indices["event"], 0] != -1
        non_zero_idx = torch.masked_select(indices["event"], mask)

        loss += loss_func(preds["event"][non_zero_idx], y["event"][non_zero_idx])

        return loss
