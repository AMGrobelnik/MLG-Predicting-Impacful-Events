import networkx as nx
import pickle
import os
import random

split_ratio = [0.8, 0.1, 0.1]
batch_size = 5
input_directory = "../../data/graphs/neighborhood_sampling/"
output_directory = "../../data/graphs/batches/"

pickle_files = os.listdir(input_directory)
random.shuffle(pickle_files)

num_batches = len(pickle_files) / batch_size

for i in range(0, len(pickle_files), batch_size):
    batch_num = i / batch_size
    
    batch = pickle_files[i: i + batch_size]

    combined_graph = nx.DiGraph()

    for file_name in batch:
        file_path = os.path.join(input_directory, file_name)
        with open(file_path, 'rb') as f:
            current_graph = pickle.load(f)
            combined_graph = nx.compose(combined_graph, current_graph)  

    if batch_num < split_ratio[0] * num_batches:
        with open(os.path.join(output_directory, "/train/{batch_num}_train.pkl"), "wb") as f:
            pickle.dump(combined_graph, f)
    elif batch_num < (split_ratio[0] + split_ratio[1]) * num_batches:
        with open(os.path.join(output_directory, "/val/{batch_num}_val.pkl"), "wb") as f:
            pickle.dump(combined_graph, f)
    else:
        with open(os.path.join(output_directory, "/test/{batch_num}_test.pkl"), "wb") as f:
            pickle.dump(combined_graph, f)
       




