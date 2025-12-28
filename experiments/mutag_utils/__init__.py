import argparse
import torch
import numpy as np
import networkx as nx
from math import ceil
from random import shuffle as random_shuffle
from ghrr_with_attention import device as tensor_device
from ghrr_with_attention.utils import not_none, commutative_cantor_pairing, CheckpointContext, cached, cached_with_dict
from ghrr_with_attention.hv_memory import get_random_hvs
from ghrr_with_attention import hv_functions
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx as to_networkx_inner
from typing import TypeGuard, Iterable, TypeVar
from dataclasses import dataclass
from pathlib import Path
from experiments import Experiment

T = TypeVar('T')

D = 10000
m = 28

""" Assumptions:
    The MUTAG dataset has 188 graphs.
    The graphs are always presented in the same order.
    The graphs' nodes have IDs.
    The nodes' IDs are consecutive and start from 0.
    There's a maximum of 28 nodes in a graph.
"""


def get_position_encodings() -> torch.Tensor:
    def get_range_tensor(upper_limit: int) -> torch.Tensor:
        return torch.tensor(tuple(range(upper_limit)), dtype=torch.int8, device=tensor_device)
    
    def get_single_tensor(n: float) -> torch.Tensor:
        return torch.tensor(n, dtype=torch.float32, device=tensor_device)
        
    v1 = get_range_tensor(m)
    n, row, col = torch.meshgrid(v1, v1, v1, indexing="ij")
    v3 = torch.where((n == row) & (n == col), get_single_tensor(1.0), get_single_tensor(0.0))
    return v3[:, None, :, :].expand(m, D, m, m)

def get_encodings(path: Path | None) -> torch.Tensor:
    return get_random_hvs(D, m, path, m, device=tensor_device)

@dataclass
class RawGraphData:
    id: int
    label: int
    graph: nx.Graph

@dataclass
class EncodedGraphData:
    id: int
    label: int
    hv: torch.Tensor

root_path = Path(__file__) / "../../_results"

dataset_path = root_path / "tudataset"
dataset_path.mkdir(parents=True, exist_ok=True)

hvs_root = root_path / "hvs"
hvs_root.mkdir(parents=True, exist_ok=True)

encodings_root = root_path / "encodings"
encodings_root.mkdir(parents=True, exist_ok=True)

def get_dataset() -> Iterable[RawGraphData]:
    def graph_is_directed(G: nx.Graph | nx.DiGraph) -> TypeGuard[nx.DiGraph]:
        return nx.is_directed(G)

    def graph_to_networkx(d: Data) -> tuple[int, nx.Graph]:
        G: nx.Graph | nx.DiGraph = to_networkx_inner(d, to_undirected=True)

        if graph_is_directed(G):
            raise ValueError("Unexpected error")

        label: int = d.y.item()
        
        return (label, G)

    dataset: Iterable[Data] = TUDataset(str(dataset_path), name="MUTAG")
    graphs: Iterable[tuple[int, nx.Graph]] = (graph_to_networkx(d) for d in dataset)
    result: Iterable[RawGraphData] = (RawGraphData(id=n, label=label, graph=g) for (n, (label, g)) in enumerate(graphs))
    return result

class Encoder:
    position_encodings: torch.Tensor
    query_encodings: torch.Tensor
    key_encodings_1: torch.Tensor
    key_encodings_2: torch.Tensor
    value_encodings: torch.Tensor
    ctx: CheckpointContext
    ctx2: CheckpointContext
    
    force_calculation: bool
    
    def __init__(self,
        force_calculation: bool = False \
    ):
        self.force_calculation = force_calculation
        
        if encodings_root is None and hvs_root is not None:
            raise ValueError("If HV tensors will be stored, encoding tensors should as well, to maintain consistency")
        
        self.position_encodings = get_position_encodings()
        
        if encodings_root is None:
            self.query_encodings = get_encodings()
            self.key_encodings_1 = get_encodings()
            self.key_encodings_2 = get_encodings()
            self.value_encodings = get_encodings()
        else:
            self.query_encodings = get_encodings(encodings_root / "query.pt")
            self.key_encodings_1 = get_encodings(encodings_root / "key1.pt")
            self.key_encodings_2 = get_encodings(encodings_root / "key2.pt")
            self.value_encodings = get_encodings(encodings_root / "value.pt")
        
        self.ctx = CheckpointContext(f"Graph - individual parts")
        self.ctx2 = CheckpointContext(f"Graph - whole graph")

    def __print(self, checkpoints: bool, msg: str):
        if checkpoints: self.ctx.print(msg)
    
    def encode(self, data: RawGraphData, *, checkpoints: bool = True) -> EncodedGraphData:
        g_id: int = data.id
        graph: nx.Graph = data.graph
        label: int = data.label
        
        self.ctx2.print(f"Graph {g_id} - Start")

        if hvs_root is not None and (hvs_root / f"{g_id}.pt").exists():
            if args.force:
                self.__print(checkpoints, f"Graph {g_id} already exists, but --force is activated. Overriding")
            else:
                self.__print(checkpoints, f"Graph {g_id} already exists, and --force not activated. Skipping")
                
                hv = torch.load(hvs_root / f"{g_id}.pt", map_location=tensor_device)
                graph_data: EncodedGraphData = EncodedGraphData(id = g_id, label = label, hv = res)
                return graph_data
        
        self.__print(checkpoints, f"Graph {g_id} - Start emptying GPU cache")
        torch.cuda.empty_cache()
        self.__print(checkpoints, f"Graph {g_id} - Finish emptying GPU cache")
        
        node_max_id = graph.number_of_nodes()
        
        self.__print(checkpoints, f"Graph {g_id} - Start calculating query")
        query_hv = hv_functions.query_from_encoded(self.position_encodings[:node_max_id], self.query_encodings[:node_max_id])
        self.__print(checkpoints, f"Graph {g_id} - Finish calculating query")
        
        self.__print(checkpoints, f"Graph {g_id} - Start calculating key")
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
        
        edges1: torch.Tensor = torch.gather(self.key_encodings_1, 0, edge_indices1[..., None, None, None].expand(-1, D, m, m))
        edges2: torch.Tensor = torch.gather(self.key_encodings_2, 0, edge_indices2[..., None, None, None].expand(-1, D, m, m))
        key_positions: torch.Tensor = torch.gather(self.position_encodings, 0, edge_indices2[..., None, None, None].expand(-1, D, m, m))
        
        key_hv = hv_functions.key_from_encoded(edges1, edges2, key_positions)
        self.__print(checkpoints, f"Graph {g_id} - Finish calculating key")
        
        self.__print(checkpoints, f"Graph {g_id} - Start calculating value")
        value_hv = hv_functions.value_from_encoded(self.position_encodings[:node_max_id], self.value_encodings[:node_max_id])
        self.__print(checkpoints, f"Graph {g_id} - Finish calculating value")
        
        self.__print(checkpoints, f"Graph {g_id} - Start calculating result")
        v1: torch.Tensor = key_hv.adjoint()

        v2: torch.Tensor = hv_functions.mult(query_hv, v1)
        del v1

        v3: torch.Tensor = v2.real
        del v2

        v4: torch.Tensor = torch.nn.functional.softmax(v3, dim=1)
        del v3

        res: torch.Tensor = hv_functions.mult(v4, value_hv)
        del v4
        self.__print(checkpoints, f"Graph {g_id} - Finish calculating result")

        if hvs_root is not None:
            self.__print(checkpoints, f"Graph {g_id} - Start saving result")
            torch.save(res, hvs_root / f"{g_id}.pt")
            self.__print(checkpoints, f"Graph {g_id} - Finish saving result")

        del query_hv
        del edge_indices1
        del edge_indices2
        del edges1
        del edges2
        del key_positions
        del key_hv
        del value_hv
        
        graph_data: EncodedGraphData = EncodedGraphData(id = g_id, label = label, hv = res)

        self.ctx2.print(f"Graph {g_id} - Finish")
        return graph_data

def get_stored_hv(self, data: RawGraphData) -> EncodedGraphData:
    g_id = data.id
    label = data.label
    
    if hvs_root is None:
        raise Exception("No encoded graph storage specified")
    
    if not (hvs_root / f"{g_id}.pt").exists():
        raise Exception(f"Requested HV for graph with id {g_id} does not exist")
    
    hv = torch.load(hvs_root / f"{g_id}.pt", map_location=tensor_device)
    graph_data: EncodedGraphData = EncodedGraphData(id = g_id, label = label, hv = res)
    return graph_data

@dataclass
class ConfusionMatrix:
    tp: int
    tn: int
    fp: int
    fn: int
    
    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp)
    
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn)

    def f1(self) -> float:
        precision = self.precision()
        recall = self.recall()
        return 2 * precision * recall / (precision + recall)
    
class Model:
    classes: dict[int, torch.Tensor] | None
    
    def __init__():
        self.classes = None
    
    def train(self, graphs: Iterable[EncodedGraphData]):
        sums: dict[int, torch.Tensor] = dict()
        
        for encoded in graphs:
            label = encoded.label
            if label not in sums:
                sums[label] = encoded.hv
            else:
                new_hv = sums[label] + encoded.hv
                del sums[label]
                del encoded.hv
                sums[label] = new_hv
        
        classes: dict[int, torch.Tensor] = dict()
        for k in tuple(sums.keys()):
            classes[k] = hv_functions.normalize(sums[k])
            del sums[k]
        
        self.classes = classes
    
    def predict(input_hv: torch.Tensor) -> int:
        if self.classes is None:
            raise Exception("Model hasn't been trained yet")
        
        closest_label: int | None = None
        min_distance: torch.float64 | None = None

        for label, class_hv in self.classes.items():
            distance: torch.float64 = hv_functions.normalized_similarity(input_hv, class_hv).item()
            
            if min_distance is None or distance < min_distance:
                min_distance = distance
                closest_label = label
        
        if not not_none(closest_label):
            raise Exception("Unexpected exception")
        
        return closest_label

def split_list_randomly(input: list[T], ratio: float) -> tuple[tuple[T, ...], tuple[T, ...]]:
    random_shuffle(input)
    pivot = ceil(ratio * len(input))
    v1 = tuple(input[:pivot])
    v2 = tuple(input[pivot:])
    return (v1, v2)

# Prepares all graph encodings in advance
def f1():
    encoder: Encoder = Encoder()

    dataset: Iterable[RawGraphData] = get_dataset()
    
    for raw in dataset:
        v = encoder.encode(raw)
        del v.hv

experiments: dict[str, Experiment] = { \
    "1": Experiment(fn = lambda _: f1()) \
}

__all__ = "experiments"