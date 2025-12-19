import argparse
import torch
import numpy as np
import networkx as nx
from ghrr_with_attention import device as tensor_device
from ghrr_with_attention.utils import not_none, commutative_cantor_pairing, CheckpointContext
from ghrr_with_attention.hv_memory import get_random_hvs
from ghrr_with_attention import hv_functions
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx as to_networkx_inner
from typing import TypeGuard, Iterable
from pathlib import Path
from experiments import Experiment

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--no-cpu", action="store_true")
    return parser

D = 10000
m = 28

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

def graph_to_networkx(d: Data) -> tuple[int, nx.Graph]:
    G: nx.Graph | nx.DiGraph = to_networkx_inner(d, to_undirected=True)

    if graph_is_directed(G):
        raise ValueError("Unexpected error")
    
    label: int = int(d.y.item())

    return (label, G)

def get_position_encodings() -> torch.Tensor:
    v1 = np.fromfunction(
        lambda n, depth, row, col: np.where((n == row) & (n == col), 1, 0),
        shape=(m, D, m, m),
        dtype=np.float32
    )
    return torch.from_numpy(v1).to(tensor_device)

def get_encodings(path: Path) -> torch.Tensor:
    return get_random_hvs(D, m, path, m, device=tensor_device)

def execute(raw_args: Iterable[str]) -> None:
    args = get_parser().parse_args(raw_args)
    root = Path(__file__).parent
    
    if args.no_cpu and tensor_device.type != "cuda":
        print("GPU not available. Exiting")
        return
    
    hvs_root = root / "hvs"
    if hvs_root.exists() and any(hvs_root.iterdir()) and not args.force:
        print("HVs already printed, and --force not activated")
        return
    hvs_root.mkdir(parents=True, exist_ok=True)
    
    dataset_root = root / "tudataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset: Iterable[Data] = TUDataset(root=str(dataset_root), name="MUTAG")
    
    graphs_with_labels = tuple(graph_to_networkx(d) for d in dataset)
    
    position_encodings = get_position_encodings()
    
    encodings_root = root / "encodings"
    encodings_root.mkdir(parents=True, exist_ok=True)
    query_encodings = get_encodings(encodings_root / "_query.pt")
    key_encodings_1 = get_encodings(encodings_root / "_key1.pt")
    key_encodings_2 = get_encodings(encodings_root / "_key2.pt")
    value_encodings = get_encodings(encodings_root / "_value.pt")
    
    ctx1 = CheckpointContext(f"Graph - individual parts")
    ctx2 = CheckpointContext(f"Graph - whole graph")
    ctx3 = CheckpointContext(f"Graph - all graphs", msg="Start")
    for g_id, (_, graph) in enumerate(graphs_with_labels):
        ctx2.print(f"Graph {g_id} - Start")
        
        node_max_id = graph.number_of_nodes()
        
        ctx1.print(f"Graph {g_id} - Start calculating query")
        query_hv = hv_functions.query_from_encoded(position_encodings[:node_max_id], query_encodings[:node_max_id])
        ctx1.print(f"Graph {g_id} - Finish calculating query")
        
        ctx1.print(f"Graph {g_id} - Start calculating value")
        value_hv = hv_functions.value_from_encoded(position_encodings[:node_max_id], value_encodings[:node_max_id])
        ctx1.print(f"Graph {g_id} - Finish calculating value")
        
        ctx1.print(f"Graph {g_id} - Start calculating key")
        edge_dict: dict[int, tuple[int, int]] = dict()
        for u, v in graph.edges:
            id = commutative_cantor_pairing(u, v)
            if id in edge_dict: continue
            edge_dict[id] = (u, v)
        
        edge_indices1: torch.Tensor = torch.empty((len(edge_dict),), dtype=torch.int32, device=tensor_device)
        edge_indices2: torch.Tensor = torch.empty((len(edge_dict),), dtype=torch.int32, device=tensor_device)
        for i, (u, v) in enumerate(edge_dict.values()):
            if u > v: (u, v) = (v, u)
            edge_indices1[i] = u
            edge_indices2[i] = v
        
        edges1: torch.Tensor = torch.gather(key_encodings_1, 0, edge_indices1[..., None, None, None])
        edges2: torch.Tensor = torch.gather(key_encodings_2, 0, edge_indices2[..., None, None, None])
        
        key_hv = hv_functions.key_from_encoded(edges1, edges2, position_encodings)
        ctx1.print(f"Graph {g_id} - Finish calculating key")
        
        ctx1.print(f"Graph {g_id} - Start calculating result")
        print(query_hv.shape)
        print(key_hv.shape)
        print(value_hv.shape)
        v1: torch.Tensor = key_hv.adjoint()
        v2: torch.Tensor = hv_functions.mult(query_hv, v1)
        v3: torch.Tensor = v2.real
        v4: torch.Tensor = torch.nn.functional.softmax(v3, dim=1)
        res: torch.Tensor = hv_functions.mult(v4, value_hv)
        ctx1.print(f"Graph {g_id} - Finish calculating result")

        ctx1.print(f"Graph {g_id} - Start saving result")
        torch.save(res, hvs_root / f"{g_id}.pt")
        ctx1.print(f"Graph {g_id} - Finish saving result")

        ctx2.print(f"Graph {g_id} - Finish")
    ctx3.print("End")

experiment: Experiment = Experiment(fn = execute)

__all__ = ["experiment"]
