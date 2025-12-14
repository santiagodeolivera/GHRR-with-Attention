import numpy as np
import networkx as nx
from ghrr_with_attention.utils import not_none
from ghrr_with_attention.hv_memory import get_random_hvs
from torch_geometric.datasets import TUDataset
from typing import TypeGuard
from pathlib import Path

D = 10000
m = 28

dataset: Iterable[Data] = TUDataset(root="tudataset", name="MUTAG")

def graph_is_directed(G: nx.Graph | nx.DiGraph) -> TypeGuard[nx.DiGraph]:
    return nx.is_directed(G)

def graph_to_networkx(d: Data) -> tuple[int, nx.Graph]:
    G: nx.Graph | nx.DiGraph = to_networkx_inner(d, to_undirected=True)

    if graph_is_directed(G):
        raise ValueError("Unexpected error")
    
    label: int = int(d.y.item())

    return (label, G)

def get_position_encodings() -> np.ndarray:
    return np.fromfunction(
        lambda n, depth, row, col: np.where((n == row) & (n == col), 1, 0),
        shape=(m, D, m, m)
    )

def get_encodings(path: Path) -> np.ndarray:
    return get_random_hvs(D, m, path, m)

def execute() -> None:
    # Obtain graphs
    graphs = tuple(graph_to_networkx(d) for d in dataset)

    # Get sets of vertices from all graphs, as a partial 2D matrix
    # Shape: (graph_index, vertex_index)
    

def get_experiment() -> Experiment:
    return Experiment(fn = execute)

__all__ = ["get_experiment"]
