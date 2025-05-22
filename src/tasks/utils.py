import json
import pickle


def load_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, file_path, indent=4):
    """Write data to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_jsonl(file_path):
    """Load data from a JSONL (JSON Lines) file. Returns a list of JSON objects."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data


def write_jsonl(data_list, file_path):
    """
    Write a list of dictionaries to a JSONL (JSON Lines) file.
    Each dictionary will be written as a separate line.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")


def load_pickle(file_path):
    """Load a Python object from a Pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def write_pickle(obj, file_path):
    """Write a Python object to a Pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
