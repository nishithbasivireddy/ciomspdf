import json
from pathlib import Path


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)
