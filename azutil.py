import json


def load_args(path):
    with open(path, 'r') as f:
        return json.load(f)
