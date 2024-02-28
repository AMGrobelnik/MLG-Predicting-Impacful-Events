import os
import pickle
import subprocess
import requests
import json
from tqdm import tqdm

# Function to get embeddings using OpenAI API
def get_embeddings(text, model, dimensions=100):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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",  # Replace with your OpenAI API key
    }
    data = {"input": text, "model": model, "dimensions": dimensions}
    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    embeddings = response_data["data"][0]["embedding"]
    return embeddings


# Load the output from the get_concept_text
output_directory = "../../data/text/"
output_file = "concepts_text.pkl"
output_path = os.path.join(output_directory, output_file)
with open(
    output_path, "rb"
) as file:  # Change "r" to "rb" to read the file in binary mode
    concept_df = pickle.load(file)  # Use pickle.load() to load the pickled file

# Load the dataset from dataset.pkl
dataset_path = "../../data/text/dataset.pkl"
with open(dataset_path, "rb") as file:
    event_df = pickle.load(file)

# Get embeddings for concept_df using the desired model
model_name = "text-embedding-3-small"  # Replace with the desired model name
# Better & more expensive: text-embedding-3-large

# truncate for testing
concept_df = concept_df[:10]
event_df = event_df[:10]

# Initialize new dataframe columns
concept_df["embedding"] = ""
event_df["summary_embedding"] = ""
event_df["title_embedding"] = ""

# Get embeddings for each label in concept_df and save them back to the same dataframe
for index, row in tqdm(concept_df.iterrows(), total=len(concept_df), desc="Embedding concepts"):
    label = row["label"]
    embedding = get_embeddings(label, model_name, dimensions=7)
    concept_df.at[index, "embedding"] = embedding

# Get embeddings for summary and title fields in event_df and add them to the dataframe
for index, row in tqdm(event_df.iterrows(), total=len(event_df), desc="Embedding events"):
    summary = row["summary"]
    title = row["title"]
    summary_embedding = get_embeddings(summary, model_name)
    title_embedding = get_embeddings(title, model_name)
    event_df.at[index, "summary_embedding"] = summary_embedding
    event_df.at[index, "title_embedding"] = title_embedding


# Save concept_df as a pickle file
concept_output_file = "concept_embeds.pkl"
concept_output_path = os.path.join(output_directory, concept_output_file)
with open(concept_output_path, "wb") as file:
    pickle.dump(concept_df, file)

# Save event_df as a pickle file
event_output_file = "event_embeds.pkl"
event_output_path = os.path.join(output_directory, event_output_file)
with open(event_output_path, "wb") as file:
    pickle.dump(event_df, file)
