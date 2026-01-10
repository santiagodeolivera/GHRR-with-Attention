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

def get_mutag_dataset(root_dir: Path) -> Iterable[nx.Graph]:
	dataset: Iterable[Data] = TUDataset(root=str(root_dir), name="MUTAG")
	graphs = tuple(graph_to_networkx(d) for d in dataset)
	return graphs

__all__ = ["get_mutag_dataset"]
