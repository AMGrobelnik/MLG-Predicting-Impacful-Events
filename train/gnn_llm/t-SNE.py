import torch
import matplotlib.pyplot as plt
from openTSNE import TSNE
import pickle
from deepsnap.hetero_graph import HeteroGraph
import networkx as nx
from hetero_gnn import HeteroGNN
from train_gnn_llm import graph_tensors_to_device


train_args = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "hidden_size": 81,
    "epochs": 233,
    "weight_decay": 0.00002203762357664057,
    "lr": 0.003873757421883433,
    "attn_size": 48,
    "num_layers": 6,
    "aggr": "attn",
}


tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
    n_iter=50,
)

# Assume node_embeddings is your PyTorch tensor of vectors
# For example: node_embeddings = torch.randn(100, 128) # 100 vectors of 128 dimensions each


def plot_tsne(node_embeddings):
    # Convert the tensor to a numpy array if it's not already
    # if isinstance(node_embeddings, torch.Tensor):
    #     node_embeddings = node_embeddings.cpu().numpy()

    # Apply t-SNE reduction

    embedding_train = tsne.fit(node_embeddings)
    embeddings_2d = embedding_train.transform(node_embeddings)

    # tsne = TSNE(n_components=2, random_state=0)
    # embeddings_2d = tsne.fit_transform(node_embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker="o", cmap="Spectral")
    plt.title("t-SNE of Node Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar()
    plt.show()


with open("./1_concepts_similar_llm.pkl", "rb") as f:
    G = pickle.load(f)

# Create a HeteroGraph object from the networkx graph
hetero_graph = HeteroGraph(G, netlib=nx, directed=True)
graph_tensors_to_device(hetero_graph)

model = HeteroGNN(
    hetero_graph,
    train_args,
    num_layers=train_args["num_layers"],
    aggr=train_args["aggr"],
    return_embedding=True,
).to(train_args["device"])

model.load_state_dict(torch.load("./best_model.pkl"))

preds = model(hetero_graph.node_feature, hetero_graph.edge_index)

print("Preds", preds["event"])
print("Shape", preds["event"].shape)

# Call the function with your embeddings
plot_tsne(preds["event"].cpu().detach().numpy())
