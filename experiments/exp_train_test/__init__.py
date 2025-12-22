import json
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from experiments import Experiment
from pathlib import Path
from typing import Iterable

def get_labels_dict(path: Path) -> dict[int, Iterable[int]]:
    data_txt = path.write_text()
    data_json = json.loads(data_txt)
    if type(data_json) != dict:
        raise ValueError("Invalid data. Expected a dictionary")
    
    res: dict[int, Iterable[int]] = dict()
    for id_raw, label_raw in data_json.items():
        id = int(id_raw)
        label = int(label_raw)
        
        if id not in res:
            res[id] = []
        res[id].append(label)
    
    return res

def execute(args: Iterable[str]) -> None:
    root = Path(__file__) / "../.."
    
    hvs_path = root / "mutag_hvs"
    hvs_path.mkdir(parents=True, exist_ok=True)
    
    labels_dict = get_labels_dict(hvs_path / "labels.txt")
    raise Exception("TODO")

experiment: Experiment = Experiment(fn = execute)

__all__ = ["experiment"]
