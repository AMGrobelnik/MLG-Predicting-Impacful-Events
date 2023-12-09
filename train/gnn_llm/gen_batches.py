import networkx as nx
import pickle
import os
import random
from tqdm import tqdm

split_ratio = [1.0, 0.0, 0.0]
batch_size = 1
input_directory = "../../data/graphs/neighborhood_sampling/"
output_directory = "../../data/graphs/batches/"

def get_nodes_by_type(graph, node_type):
    return [node for node, data in graph.nodes(data=True) if data.get('node_type') == node_type]


pickle_files = os.listdir(input_directory)
random.shuffle(pickle_files)

num_batches = len(pickle_files) / batch_size

# Create output directories if they don't exist
for sub_dir in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_directory, sub_dir), exist_ok=True)


for i in tqdm(range(0, len(pickle_files), batch_size), desc="Processing Batches"):
    batch_num = int(i / batch_size)

    batch = pickle_files[i : i + batch_size]

    combined_graph = nx.DiGraph()
    # TODO: mnozica A in B, pazi pri zdruzevanju
    for file_name in batch:
        file_path = os.path.join(input_directory, file_name)
        with open(file_path, "rb") as f:
            current_graph = pickle.load(f)
            combined_graph = nx.compose(combined_graph, current_graph)
            # Choose 10 random nodes of type 'event'
            event_nodes = get_nodes_by_type(combined_graph, 'event')
            selected_nodes = random.sample(event_nodes, min(10, len(event_nodes)))
            for node in selected_nodes:
                combined_graph.nodes[node]['node_type'] = 'event_target'
            print(combined_graph.number_of_nodes())

    # if os.path.exists(os.path.join(output_directory, "/train")) == False:
    #     os.mkdir(os.path.join(output_directory, "/train"))
    # if os.path.exists(os.path.join(output_directory, "/val")) == False:
    #     os.mkdir(os.path.join(output_directory, "/val"))
    # if os.path.exists(os.path.join(output_directory, "/test")) == False:
    #     os.mkdir(os.path.join(output_directory, "/test"))

    if batch_num < split_ratio[0] * num_batches:
        with open(
            os.path.join(output_directory, f"train/{batch_num}_train.pkl"), "wb"
            os.path.join(output_directory, f"train/{batch_num}_train.pkl"), "wb"
        ) as f:
            pickle.dump(combined_graph, f)
    elif batch_num < (split_ratio[0] + split_ratio[1]) * num_batches:
        with open(
            os.path.join(output_directory, f"val/{batch_num}_val.pkl"), "wb"
            os.path.join(output_directory, f"val/{batch_num}_val.pkl"), "wb"
        ) as f:
            pickle.dump(combined_graph, f)
    else:
        with open(
            os.path.join(output_directory, f"test/{batch_num}_test.pkl"), "wb"
            os.path.join(output_directory, f"test/{batch_num}_test.pkl"), "wb"
        ) as f:
            pickle.dump(combined_graph, f)
