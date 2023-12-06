import pickle
import networkx as nx
import walker
import random

batch_size=2
walks_per_node= 2
k= 2

# Load the heterogeneous graph data
with open("train/gnn_llm/1_concepts_similar_llm.pkl", "rb") as f:
    G = pickle.load(f)
    
sampled_events = {}
sampled_nodes = random.sample(G.nodes, k)
sampled_graph = G.subgraph(sampled_nodes)

X = walker.random_walks(G, n_walks=walks_per_node, walk_len=k, start_nodes=[0, 1, 2])

print(X)