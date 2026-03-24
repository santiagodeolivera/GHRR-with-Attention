from pathlib import Path
import itertools
from typing import Callable

import networkx as nx
import torch

from hv_functions import UpperTensorFunctionsManager
from hv_memory import get_random_hvs
from utils import get_range_tensor, commutative_cantor_pairing, Timer
from tudataset import get_dataset_main, get_graph_dataset
from constants import D, m
from fs_organization import FsOrganizer
from gpu_management import TensorProxy, TensorFunctionsManager, DataType
from fn_context import FnContext

def get_position_encodings(manager: TensorFunctionsManager, out: Path) -> torch.Tensor:
    max_num_nodes = get_dataset_main().max_num_nodes
    
    const0 = torch.tensor(0.0, dtype=torch.float32)
    const1 = torch.tensor(1.0, dtype=torch.float32)
    
    def f1(dims: tuple[torch.Tensor, ...]) -> torch.Tensor:
        n, row, col = dims
        return torch.where(row == col, \
            torch.clamp(n * (m / max_num_nodes) - row, const0, const1), \
        const0).type(torch.complex64)
    
    mid_result = manager.new_from_function((max_num_nodes, m, m), DataType.complex64, f1, out=out)
    position_encodings = mid_result.tensor()[:, None, :, :].expand(max_num_nodes, D, m, m)
    
    return position_encodings

def create_and_save_hv( \
    g_id: int, \
    graph: nx.Graph, \
    out_path: Path, \
    position_encodings: torch.Tensor, \
    query_encodings: TensorProxy, \
    key_encodings_1: TensorProxy, \
    key_encodings_2: TensorProxy, \
    value_encodings: TensorProxy, \
    functions: UpperTensorFunctionsManager \
) -> TensorProxy:
    tmp = functions.tmp_gen.new_paths(3)
    node_max_id = graph.number_of_nodes()
    
    timer = Timer("Calculate Query HV")
    query_hv = functions.query_from_encoded(position_encodings[:node_max_id], query_encodings.tensor()[:node_max_id], \
        out=tmp[0])
    timer.end()
    
    timer = Timer("Calculate Value HV")
    value_hv = functions.value_from_encoded(position_encodings[:node_max_id], value_encodings.tensor()[:node_max_id], \
        out=tmp[1])
    timer.end()
    
    timer = Timer("Calculate Key HV")
    edge_dict: dict[int, tuple[int, int]] = dict()
    for u, v in graph.edges:
        id = commutative_cantor_pairing(u, v)
        if id in edge_dict: continue
        edge_dict[id] = (u, v)
    
    edge_indices1: torch.Tensor = torch.empty((len(edge_dict),), dtype=torch.int32)
    edge_indices2: torch.Tensor = torch.empty((len(edge_dict),), dtype=torch.int32)
    for i, (u, v) in enumerate(edge_dict.values()):
        if u > v: (u, v) = (v, u)
        edge_indices1[i] = u
        edge_indices2[i] = v
    
    edges1: torch.Tensor = torch.gather(key_encodings_1.tensor(), 0, \
        edge_indices1[..., None, None, None].expand(-1, D, m, m))
    edges2: torch.Tensor = torch.gather(key_encodings_2.tensor(), 0, \
        edge_indices2[..., None, None, None].expand(-1, D, m, m))
    # torch.cuda.empty_cache()
    key_positions: torch.Tensor = torch.gather(position_encodings, 0, \
        edge_indices2[..., None, None, None].expand(-1, D, m, m))
    
    key_hv = functions.key_from_encoded(edges1, edges2, key_positions, out=tmp[2])
    timer.end()
    del edge_indices1
    del edge_indices2
    del edges1
    del edges2
    del key_positions
    # torch.cuda.empty_cache()
    
    timer = Timer("Calculate Attention HV")
    res = functions.attention_function(query_hv.tensor(), key_hv.tensor(), value_hv.tensor(), out=out_path)
    timer.end()
    del query_hv
    del key_hv
    del value_hv
    
    return res

def action_create_hv(g_id: int, ctx: FnContext) -> None:
    root = ctx.fs
    functions = ctx.functions
    max_num_nodes = get_dataset_main().max_num_nodes
    
    query_encodings = get_random_hvs(functions.lower, root.query_encodings, max_num_nodes)
    key_encodings_1 = get_random_hvs(functions.lower, root.key_encodings_1, max_num_nodes)
    key_encodings_2 = get_random_hvs(functions.lower, root.key_encodings_2, max_num_nodes)
    value_encodings = get_random_hvs(functions.lower, root.value_encodings, max_num_nodes)
    
    position_encodings = get_position_encodings(functions.lower, functions.tmp_gen.new_paths(1)[0])
    
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

