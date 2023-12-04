import torch
import deepsnap
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_sparse import matmul

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
        for (src, edge_type, dst), item in message_type_emb.items():
            node_emb[dst].append(item)

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
            s = self.attn_proj(xs).squeeze(
                -1
            )  # Pass the xs through the attention layer
            s = torch.mean(
                s, dim=-1
            )  # Average the attention scores across the source nodes
            self.alpha = torch.softmax(
                s, dim=0
            ).detach()  # Compute the attention probability
            out = self.alpha.reshape(-1, 1, 1) * xs
            out = torch.sum(out, dim=0)
            return out
        
        raise ValueError(f"Invalid aggr {self.aggr}, valid options: (mean, attn)")


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
    def __init__(self, hetero_graph, args, num_layers, aggr="mean", return_embedding=False):
        super(HeteroGNN, self).__init__()

        self.aggr = aggr
        self.hidden_size = args["hidden_size"]
        self.num_layers = num_layers
        self.return_embedding = return_embedding

        # Use a single ModuleDict for batch normalization and ReLU layers
        self.bns = nn.ModuleDict()
        self.relus = nn.ModuleDict()
        self.convs = nn.ModuleList()
        self.fc = nn.ModuleDict()  # Prediction heads

        # Initialize graph convolutional layers for each layer and message type
        for i in range(self.num_layers):
            first_layer = i == 0
            conv = HeteroGNNWrapperConv(
                    generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer),
                    args,
                    self.aggr)
            self.convs.append(conv)

        # Initialize batch normalization and ReLU layers for each layer and node type
        all_node_types = hetero_graph.node_types
        for i in range(self.num_layers):
            for node_type in all_node_types:
                key_bn = f"bn_{i}_{node_type}"
                key_relu = f"relu_{i}_{node_type}"
                self.bns[key_bn] = nn.BatchNorm1d(self.hidden_size, eps=1.0)
                self.relus[key_relu] = nn.LeakyReLU()

        # Initialize fully connected layers for each node type
        for node_type in all_node_types:
            self.fc[node_type] = nn.Linear(self.hidden_size, 1)

    def forward(self, node_feature, edge_index):
        """
        Forward pass of the model.

        :param node_feature: Dictionary of node features for each node type.
        :param edge_index: Dictionary of edge indices for each message type.
        :return: The output embeddings for each node type after passing through the model.
        """
        x = node_feature

        # Apply graph convolutional, batch normalization, and ReLU layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)  # Apply the i-th graph convolutional layer
            for node_type in x:
                key_bn = f"bn_{i}_{node_type}"
                key_relu = f"relu_{i}_{node_type}"
                x[node_type] = self.bns[key_bn](
                    x[node_type]
                )  # Apply batch normalization
                x[node_type] = self.relus[key_relu](x[node_type])  # Apply ReLU

        if self.return_embedding:
            return x

        # Apply the prediction head (linear layer)
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

        # mape = MeanAbsolutePercentageError().to(train_args["device"])

        loss = 0
        loss_func = torch.nn.MSELoss()

        # loss_func = mape
        # MAPE PRODUCES BETTER EVAL RESULTS BUT WORSE PREDICTIONS

        mask = y["event"][indices["event"], 0] != -1
        non_zero_idx = torch.masked_select(indices["event"], mask)

        loss += loss_func(preds["event"][non_zero_idx], y["event"][non_zero_idx])

        return loss
