from pathlib import Path
import itertools
from typing import Callable

import networkx as nx
import torch

from hv_functions import UpperTensorFunctionsManager
from hv_memory import get_random_hvs
from utils import get_range_tensor, commutative_cantor_pairing
from time_ import Timer
from tudataset import get_dataset_main, get_graph_dataset
from constants import D, m
from fs_organization import FsOrganizer
from gpu_management.tensor_functions import TensorFunctionsManager
from gpu_management.data_type import DataType
from fn_context import FnContext
from mmap_tensors import MmapTensors

position_encodings_cache: torch.Tensor | None = None
def get_position_encodings(manager: TensorFunctionsManager) -> torch.Tensor:
    global position_encodings_cache
    if position_encodings_cache is not None:
        return position_encodings_cache
    
    max_num_nodes = get_dataset_main().max_num_nodes
    
    const0 = torch.tensor(0.0, dtype=torch.float32)
    const1 = torch.tensor(1.0, dtype=torch.float32)
    
    def f1(dims: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        n, row, col = dims
        
        result = torch.where(row == col, \
            torch.clamp(n * (m / max_num_nodes) - row, const0, const1), \
        const0)
        
        out[...] = result.type(torch.complex64)
    
    mid_result = manager.new_from_function((max_num_nodes, m, m), DataType.complex64, f1)
    position_encodings = mid_result[:, None, :, :].expand(max_num_nodes, D, m, m)
    
    position_encodings_cache = position_encodings
    return position_encodings

def create_and_save_hv( \
    g_id: int, \
    graph: nx.Graph, \
    out_path: Path, \
    position_encodings: torch.Tensor, \
    query_encodings: torch.Tensor, \
    key_encodings_1: torch.Tensor, \
    key_encodings_2: torch.Tensor, \
    value_encodings: torch.Tensor, \
    functions: UpperTensorFunctionsManager \
) -> None:
    node_max_id = graph.number_of_nodes()
    
    timer = Timer("Calculate Query HV")
    query_hv = functions.query_from_encoded(position_encodings[:node_max_id], query_encodings[:node_max_id])
    timer.end()
    
    timer = Timer("Calculate Value HV")
    value_hv = functions.value_from_encoded(position_encodings[:node_max_id], value_encodings[:node_max_id])
    timer.end()
    
    timer = Timer("Calculate Key HV")
    edge_dict: dict[int, tuple[int, int]] = dict()
    for u, v in graph.edges:
        id = commutative_cantor_pairing(u, v)
        if id in edge_dict: continue
        edge_dict[id] = (u, v)
    
    edge_dict_len = len(edge_dict)
    edge_indices1: torch.Tensor = torch.empty((edge_dict_len,), dtype=torch.int32)
    edge_indices2: torch.Tensor = torch.empty((edge_dict_len,), dtype=torch.int32)
    del edge_dict_len
    for i, (u, v) in enumerate(edge_dict.values()):
        if u > v: (u, v) = (v, u)
        edge_indices1[i] = u
        edge_indices2[i] = v
    del edge_dict
    
    edge_indices1_expanded = edge_indices1[..., None, None, None].expand(-1, D, m, m)
    edge_indices2_expanded = edge_indices2[..., None, None, None].expand(-1, D, m, m)
    edges1: torch.Tensor = torch.gather(key_encodings_1, 0, edge_indices1_expanded)
    edges2: torch.Tensor = torch.gather(key_encodings_2, 0, edge_indices2_expanded)
    key_positions: torch.Tensor = torch.gather(position_encodings, 0, edge_indices2_expanded)
    del edge_indices1_expanded
    del edge_indices2_expanded
    
    key_hv = functions.key_from_encoded(edges1, edges2, key_positions)
    timer.end()
    del edge_indices1
    del edge_indices2
    del edges1
    del edges2
    del key_positions
    
    timer = Timer("Calculate Attention HV")
    res = MmapTensors.new_override(out_path, (D, m, m), DataType.complex64)
    functions.attention_function(query_hv, key_hv, value_hv, out=res)
    timer.end()
    del query_hv
    del key_hv
    del value_hv

def action_create_hv(g_id: int, ctx: FnContext) -> None:
    root = ctx.fs
    functions = ctx.functions
    max_num_nodes = get_dataset_main().max_num_nodes
    
    query_encodings = get_random_hvs(functions.lower, root.query_encodings, max_num_nodes)
    key_encodings_1 = get_random_hvs(functions.lower, root.key_encodings_1, max_num_nodes)
    key_encodings_2 = get_random_hvs(functions.lower, root.key_encodings_2, max_num_nodes)
    value_encodings = get_random_hvs(functions.lower, root.value_encodings, max_num_nodes)
    
    position_encodings = get_position_encodings(functions.lower)
    
    graphs = get_graph_dataset(root.tudataset)
    graph = next(itertools.islice(graphs, g_id, None))

    create_and_save_hv( \
        g_id = g_id, \
        graph = graph, \
        out_path = root.hv_encoding_of(g_id), \
        position_encodings = position_encodings, \
        query_encodings = query_encodings, \
        key_encodings_1 = key_encodings_1, \
        key_encodings_2 = key_encodings_2, \
        value_encodings = value_encodings, \
        functions = functions \
    )

__all__ = ["action_create_hv"]

