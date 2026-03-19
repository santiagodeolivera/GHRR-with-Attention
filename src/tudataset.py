from typing import TypeGuard
from pathlib import Path
import json
from dataclasses import dataclass

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx as to_networkx_inner
from typing import Iterable
import networkx as nx

"""
Assumptions:
	The graphs are always presented in the same order.
	The graphs' nodes have IDs.
	The nodes' IDs are consecutive and start from 0.
"""

def graph_is_directed(G: nx.Graph | nx.DiGraph) -> TypeGuard[nx.DiGraph]:
	return nx.is_directed(G)

def graph_to_networkx(d: Data) -> nx.Graph:
	G: nx.Graph | nx.DiGraph = to_networkx_inner(d, to_undirected=True)

	if graph_is_directed(G):
		raise ValueError("Unexpected error")

	return G

@dataclass
class DatasetInfo:
    name: str
    num_graphs: int
    max_num_nodes: int
    
dataset_main_info: DatasetInfo | None = None

dataset_info_list: tuple[DatasetInfo, ...] = (
    DatasetInfo("MUTAG", 188, 28),
    DatasetInfo("PTC_FM", 349, 64)
)

def set_dataset_main(name: str):
    global dataset_main_info
    
    if dataset_main_info is not None:
        raise Exception()
    
    try:
        dataset_main_info = next(x for x in dataset_info_list if x.name == name)
    except StopIteration:
        raise ValueError(f"Unknown dataset name: {repr(name)}")

def get_dataset_main() -> DatasetInfo:
    global dataset_main_info
    if dataset_main_info is None:
        raise Exception()
    return dataset_main_info

dataset_cache: Iterable[Data] | None = None
def get_dataset(root_dir: Path) -> Iterable[Data]:
    global dataset_cache
    
    if dataset_cache is not None:
	    return dataset_cache
    
    dataset: Iterable[Data] = TUDataset(root=str(root_dir), name=get_dataset_main().name)
    dataset_cache = dataset
    return dataset

def get_graph_dataset(tudataset_dir: Path) -> Iterable[nx.Graph]:
    dataset: Iterable[Data] = get_dataset(tudataset_dir)
    graphs = tuple(graph_to_networkx(d) for d in dataset)
    return graphs

def get_mutag_dataset_labels(tudataset_dir: Path) -> Iterable[int]:
    dataset: Iterable[Data] = get_dataset(tudataset_dir)
    ids = tuple(d.y.item() for d in dataset)
    return ids

def define_ids_to_labels_mapping(tudataset_dir: Path, out_file: Path) -> None:
	ids = tuple(get_mutag_dataset_labels(tudataset_dir))
	json_data = json.dumps(ids)
	out_file.write_text(json_data)

def get_ids_to_labels_mapping(file: Path) -> tuple[int, ...]:
	json_data = file.read_text()
	ids = json.loads(json_data)
	
	if type(ids) != list:
		raise Exception()
	
	if any(type(label) != int for label in ids):
		raise Exception()
	
	return tuple(ids)

__all__ = [
    "set_dataset_main",
    "get_dataset_main",
    
    "get_graph_dataset",
    "define_ids_to_labels_mapping",
    "get_ids_to_labels_mapping"
]

