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
    max_num_edges: int | None

def get_dataset_template(name: str) -> DatasetTemplate:
    sections = name.split(".")
    
    dataset_name = sections[0]
    if dataset_name not in ("MUTAG", "ENZYMES", "PTC_FM", "AIDS", "Letter-low", "Letter-med", "Letter-high"):
        raise Exception(f"Invalid TUDataset dataset name: {dataset_name}")
    
    pruning_flags: list[PruningMode] = []
    max_num_nodes: int | None = None
    max_num_edges: int | None = None
    
    for s in sections[1:]:
        if s == "d1":
            pruning_flags.append(PruningMode.REMOVE_DEGREE_1)
        elif s == "pruned":
            pruning_flags.append(PruningMode.PRUNE_PATHS)
        elif s.startswith("nodes"):
            max_num_nodes = int(s[5:])
        elif s.startswith("edges"):
            max_num_edges = int(s[5:])
    
    return DatasetTemplate(
        dataset_name = dataset_name,
        pruning_flags = PruningMode.from_iter(pruning_flags),
        max_num_nodes = max_num_nodes,
        max_num_edges = max_num_edges
    )

@dataclass
class DatasetInfo:
    name: str
    num_graphs: int
    max_num_nodes: int
    max_num_edges: int
    label_num: dict[int, int]

def get_label_num(labels: Iterable[int]) -> dict[int, int]:
    result: dict[int, int] = dict()
    for label in labels:
        if label not in result:
            result[label] = 0
        
        result[label] += 1
    
    return result

dataset_main_info: DatasetInfo | None = None
dataset_cache: tuple[tuple[int, nx.Graph], ...] | None = None

def set_dataset(name: str, root_dir: Path):
    global dataset_main_info
    global dataset_cache
    
    if dataset_main_info is not None:
        raise Exception()
    
    try:
        template = get_dataset_template(name)
    except Exception as e:
        raise Exception(f"Failed to load dataset {repr(name)}") from e
    
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
    
    max_num_edges = template.max_num_edges
    if max_num_edges is not None:
        graphs = ((y, G) for y, G in graphs if G.number_of_edges() <= max_num_edges)
    
    dataset_cache = tuple(graphs)
    dataset_main_info = DatasetInfo(
        name = name,
        num_graphs = len(dataset_cache),
        max_num_nodes = max(G.number_of_nodes() for y, G in dataset_cache),
        max_num_edges = max(G.number_of_edges() for y, G in dataset_cache),
        label_num = get_label_num(y for y, G in dataset_cache)
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

