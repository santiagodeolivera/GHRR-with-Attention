from typing import TypeGuard
from pathlib import Path
import json

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx as to_networkx_inner
from typing import Iterable
import networkx as nx

"""
Assumptions:
	The MUTAG dataset has 188 graphs.
	The graphs are always presented in the same order.
	The graphs' nodes have IDs.
	The nodes' IDs are consecutive and start from 0.
	There's a maximum of 28 nodes in a graph.
"""

def graph_is_directed(G: nx.Graph | nx.DiGraph) -> TypeGuard[nx.DiGraph]:
	return nx.is_directed(G)

def graph_to_networkx(d: Data) -> nx.Graph:
	G: nx.Graph | nx.DiGraph = to_networkx_inner(d, to_undirected=True)

	if graph_is_directed(G):
		raise ValueError("Unexpected error")

	return G

dataset_cache: Iterable[Data] | None = None
def get_dataset(root_dir: Path) -> Iterable[Data]:
	global dataset_cache
	
	if dataset_cache is not None:
		return dataset_cache
	
	dataset: Iterable[Data] = TUDataset(root=str(root_dir), name="MUTAG")
	dataset_cache = dataset
	return dataset

def get_mutag_dataset(tudataset_dir: Path) -> Iterable[nx.Graph]:
	dataset: Iterable[Data] = get_dataset(tudataset_dir)
	graphs = tuple(graph_to_networkx(d) for d in dataset)
	return graphs

def get_mutag_dataset_labels(tudataset_dir: Path) -> Iterable[int]:
	dataset: Iterable[Data] = get_dataset(tudataset_dir)
	ids = tuple(d.y.item() for d in dataset)
	return ids

def define_ids_to_labels_mapping(tudataset_dir: Path, out_file: Path):
	ids = dict((i, label) for (i, label) in enumerate(get_mutag_dataset_labels(tudataset_dir)))
	json_data = json.dumps(ids)
	out_file.write_text(json_data)


def get_ids_to_labels_mapping(file: Path) -> dict[int, int]:
	json_data = file.read_text()
	ids = json.loads(json_data)
	
	if type(ids) != dict:
		raise Exception()
	
	if any(type(id) != int for id in ids.keys()):
		raise Exception()
	
	if any(type(label) != int for label in ids.values()):
		raise Exception()
	
	return ids

__all__ = ["get_mutag_dataset", "define_ids_to_labels_mapping", "get_ids_to_labels_mapping"]
