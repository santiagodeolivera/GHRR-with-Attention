import torch
import numpy as np
import networkx as nx
from ghrr_with_attention import device as tensor_device
from ghrr_with_attention.utils import not_none
from ghrr_with_attention.hv_memory import get_random_hvs
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx as to_networkx_inner
from typing import TypeGuard, Iterable
from pathlib import Path
from experiments import Experiment

D = 10000
m = 28

"""
Assumptions:
    The MUTAG dataset has 188 graphs.
    The graphs' nodes have IDs.
    The nodes' IDs are consecutive and start from 0.
    There's a maximum of 28 nodes in a graph.
"""

dataset: Iterable[Data] = TUDataset(root="tudataset", name="MUTAG")

def graph_is_directed(G: nx.Graph | nx.DiGraph) -> TypeGuard[nx.DiGraph]:
    return nx.is_directed(G)

def graph_to_networkx(d: Data) -> tuple[int, nx.Graph]:
    G: nx.Graph | nx.DiGraph = to_networkx_inner(d, to_undirected=True)

    if graph_is_directed(G):
        raise ValueError("Unexpected error")
    
    label: int = int(d.y.item())

    return (label, G)

def get_position_encodings() -> torch.Tensor:
    v1 = np.fromfunction(
        lambda n, depth, row, col: np.where((n == row) & (n == col), 1, 0),
        shape=(m, D, m, m)
    )
    return torch.from_numpy(v1).to(tensor_device)

def get_encodings(path: Path) -> torch.Tensor:
    return get_random_hvs(D, m, path, m)

def execute() -> None:
    # Obtain graphs
    graphs_with_labels = tuple(graph_to_networkx(d) for d in dataset)
    labels = tuple(a for (a, b) in graphs_with_labels)
    graphs = tuple(b for (a, b) in graphs_with_labels)

    # Get sets of vertices from all graphs, as a partial 2D matrix
    graph_node_numbers = torch.tensor(tuple(g.number_of_nodes() for g in graphs))[..., None]
    v1 = torch.tensor(tuple(range(28)))
    v2 = torch.tensor(-1)
    nodes = torch.where(v1 < graph_node_numbers, v1, v2)
    nodes_ = nodes[..., None, None, None].expand(-1, -1, D, m, m)
    
    key_encodings = get_encodings(Path("key.pt"))
    
    key_encoded_nodes = torch.where(nodes_ >= 0, key_encodings.gather(0, nodes_), torch.tensor(0))
    
    # Shape: (graph_index, vertex_index)

experiment: Experiment = Experiment(fn = execute)

__all__ = ["experiment"]
