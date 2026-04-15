from typing import TypeGuard, Callable, Iterable
from pathlib import Path
import json
from dataclasses import dataclass

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx as to_networkx_inner
import networkx as nx

from time_ import Timer
from graph_pruning import PruningMode, prune_graph

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
class DatasetTemplate:
    dataset_name: str
    pruning_flags: PruningMode | None
    max_num_nodes: int | None

dataset_templates: dict[str, DatasetTemplate] = dict()
for name in ("MUTAG", "ENZYMES", "PTC_FM"):
    dataset_templates[name] = DatasetTemplate(name, None, None)
    dataset_templates[f"{name}_pruned"] = DatasetTemplate(name, PruningMode.PRUNE_PATHS, None)
    dataset_templates[f"{name}_d1"] = DatasetTemplate(name, PruningMode.REMOVE_DEGREE_1, None)
    dataset_templates[f"{name}_d1_pruned"] = DatasetTemplate(name, PruningMode.REMOVE_DEGREE_1 | PruningMode.PRUNE_PATHS, None)
    
    for max_nodes in (28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100):
        dataset_templates[f"{name}_nodes_{max_nodes}"] = \
            DatasetTemplate(name, None, max_nodes)
        
        dataset_templates[f"{name}_pruned_nodes_{max_nodes}"] = \
            DatasetTemplate(name, PruningMode.PRUNE_PATHS, max_nodes)
        
        dataset_templates[f"{name}_d1_nodes_{max_nodes}"] = \
            DatasetTemplate(name, PruningMode.REMOVE_DEGREE_1, max_nodes)
        
        dataset_templates[f"{name}_d1_pruned_nodes_{max_nodes}"] = \
            DatasetTemplate(name, PruningMode.REMOVE_DEGREE_1 | PruningMode.PRUNE_PATHS, max_nodes)

@dataclass
class DatasetInfo:
    name: str
    num_graphs: int
    max_num_nodes: int
    
dataset_main_info: DatasetInfo | None = None
dataset_cache: tuple[tuple[int, nx.Graph], ...] | None = None

def set_dataset(name: str, root_dir: Path):
    global dataset_main_info
    global dataset_cache
    
    if dataset_main_info is not None:
        raise Exception()
    
    template = dataset_templates.get(name, None)
    if template is None:
        raise ValueError(f"Unknown dataset name: {repr(name)}")
    
    name = template.dataset_name
    timer = Timer(f"Load TUDataset dataset {repr(name)}")
    dataset: Iterable[Data] = TUDataset(root=str(root_dir), name=name)
    timer.end()
    
    graphs: Iterable[tuple[int, nx.Graph]] = ((d.y.item(), graph_to_networkx(d)) for d in dataset)
    
    pruning_flags = template.pruning_flags
    if pruning_flags is not None:
        graphs = ((y, prune_graph(G, pruning_flags)) for y, G in graphs)
    
    max_num_nodes = template.max_num_nodes
    if max_num_nodes is not None:
        graphs = ((y, G) for y, G in graphs if G.number_of_nodes() <= max_num_nodes)
    
    dataset_cache = tuple(graphs)
    dataset_main_info = DatasetInfo(
        name = name,
        num_graphs = len(dataset_cache),
        max_num_nodes = max(G.number_of_nodes() for y, G in dataset_cache)
    )

def get_dataset() -> tuple[tuple[int, nx.Graph], ...]:
    global dataset_cache
    if dataset_cache is None:
        raise Exception()
    return dataset_cache

def get_dataset_info() -> DatasetInfo:
    global dataset_main_info
    if dataset_main_info is None:
        raise Exception()
    return dataset_main_info

def get_graph_dataset(tudataset_dir: Path) -> Iterable[nx.Graph]:
    dataset = get_dataset()
    return tuple(G for y, G in dataset)

def get_mutag_dataset_labels(tudataset_dir: Path) -> Iterable[int]:
    dataset = get_dataset()
    return tuple(y for y, G in dataset)

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
    "set_dataset",
    "get_dataset_info",
    
    "get_graph_dataset",
    "define_ids_to_labels_mapping",
    "get_ids_to_labels_mapping"
]

