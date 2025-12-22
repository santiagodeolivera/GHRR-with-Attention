import json
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from experiments import Experiment
from pathlib import Path

def execute(raw_args: Iterable[str]) -> None:
    root = Path(__file__) / "../.."
    
    file_path = root / "mutag_hvs/labels.txt"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    
    dataset_root = root / "tudataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset: Iterable[Data] = TUDataset(root=str(dataset_root), name="MUTAG")
    
    labels_tuple: tuple[int, ...] = tuple(int(d.y.item()) for d in dataset)
    labels_dict: dict[int, int] = dict((id, label) for (id, label) in enumerate(labels_tuple))
    labels_json: str = json.dumps(labels_dict)
    file_path.write_text(labels_json)

experiment: Experiment = Experiment(fn = execute)

__all__ = ["experiment"]
