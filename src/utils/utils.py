import json


def read_json(json_file_path: str) -> dict:
    with open(json_file_path) as f:
        f = json.load(f)
    return f


def map_class(class_prob: float, benchmark: float) -> int:
    return int(class_prob >= benchmark)
