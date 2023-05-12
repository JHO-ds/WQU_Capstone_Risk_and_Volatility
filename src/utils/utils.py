import json
import pickle
import os
from config import properties as p


def read_json(json_file_path: str) -> dict:
    with open(json_file_path) as f:
        f = json.load(f)
    return f


def load_model(pkl_file_path: str):
    return pickle.load(open(pkl_file_path, "rb"))


def map_class(class_prob: float, benchmark: float) -> int:
    return int(class_prob >= benchmark)


def make_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def dir_management():
    make_dir(p.output_folder)
    make_dir(p.market_data_output_folder)
    make_dir(p.model_prediction_path)
    make_dir(p.model_evaluation_report_path)
    make_dir(p.backtest_recent_path)
    make_dir(p.backtest_stress_path)
