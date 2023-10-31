import pandas as pd
import os
import json


def load_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            json_content = json.load(file)
            return json_content

    except Exception as e:
        print(f"Error loading '{file_path}': {str(e)}")


def save_df_to_pickle(dataframe, output_dir, file_name):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, f"{file_name}.pkl")

    try:
        dataframe.to_pickle(file_path)
    except Exception as e:
        print(f"Error saving DataFrame to '{file_path}': {str(e)}")


def filter_json(json_content):

    filtered_events = []
    for event in json_content:

        similar_events = event.get('similarEvents', {}).get('similarEvents', [])
        concepts = event.get('info', {}).get('concepts', [])
        article_counts_total = event.get('info', {}).get('articleCounts', {}).get('total', 0)

        if similar_events or concepts or article_counts_total > 0:
            event['similarEvents'] = event['similarEvents']['similarEvents']
            filtered_events.append(event)

        event['info'].pop('eventDateEnd', None)
        event['info'].pop('categories', None)
    
    dataframe = pd.DataFrame(filtered_events)
    return dataframe

directory_path = "./data/source"
output_dir = "./data/preprocessed"
for filename in sorted(os.listdir(directory_path)):
    file_path = os.path.join(directory_path, filename)
    if file_path.endswith(".json"):
        json_content = load_json_file(file_path)
        dataframe = filter_json(json_content)

        file_name = os.path.splitext(filename)[0]

        save_df_to_pickle(dataframe, output_dir, file_name)
        i += 1
    

