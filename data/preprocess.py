import json
import os
from datetime import datetime as dt

import pandas as pd
from tqdm import tqdm


def generate_id(orig_id: str, prefix: str) -> str:
    """
    Generates an ID for a given concept or event
    :param orig_id: the original ID of the concept or event
    :param prefix: the prefix to be added to the ID (e.g. 'c' for concept, 'e' for event)
    """
    return f"{prefix}_{orig_id}"


def event_date_to_timestamp(event: dict, similar_event: bool):
    event_date = event["eventDate"] if similar_event else event["info"]["eventDate"]
    if event_date == "":
        return 0
    return round(dt.strptime(event_date, "%Y-%m-%d").timestamp() / 3600 / 24)


def load_json_file(file_path):
    try:
        with open(file_path, "r") as file:
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
        # generate unique ids
        event["info"]["uri"] = generate_id(event["info"]["uri"], "e")

        similar_events = event.get("similarEvents", {}).get("similarEvents", [])
        concepts = event.get("info", {}).get("concepts", [])
        article_counts_total = (
            event.get("info", {}).get("articleCounts", {}).get("total", 0)
        )
        event["info"]["eventDate"] = event_date_to_timestamp(event, False)

        if similar_events or concepts or article_counts_total > 0:
            # fix json structure
            event["similarEvents"] = event["similarEvents"]["similarEvents"]

            for se in similar_events:
                # generate unique ids
                se["uri"] = generate_id(se["uri"], "e")
                # convert to timestamp
                se["eventDate"] = event_date_to_timestamp(se, True)

            filtered_events.append(event)

            for c in concepts:
                # generate unique ids
                c["id"] = generate_id(c["id"], "c")

        event["info"].pop("eventDateEnd", None)
        event["info"].pop("categories", None)

    dataframe = pd.DataFrame(filtered_events)
    return dataframe


if __name__ == "__main__":
    directory_path = "./source"
    output_dir = "./preprocessed"

    files = sorted(os.listdir(directory_path))
    files = [filename for filename in files if filename.endswith(".json")]

    files = files[:201]
    for filename in tqdm(files, ncols=100, desc="Processing"):
        file_path = os.path.join(directory_path, filename)

        json_content = load_json_file(file_path)
        dataframe = filter_json(json_content)

        file_name = os.path.splitext(filename)[0]

        save_df_to_pickle(dataframe, output_dir, file_name)
