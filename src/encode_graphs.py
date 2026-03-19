from pathlib import Path
import itertools
from typing import Callable

import networkx as nx
import torch

import hv_functions
from hv_memory import get_complex_random_hvs
from utils import ICheckpointContext, CheckpointContext, VoidCheckpointContext, get_range_tensor, commutative_cantor_pairing
from device import default_device
from tudataset import get_dataset_main, get_graph_dataset
from constants import D, m
from fs_organization import FsOrganizer
import localTypes

def get_position_encodings() -> torch.Tensor:
	max_num_nodes = get_dataset_main().max_num_nodes
	
	v1 = get_range_tensor(m)
	v2 = get_range_tensor(max_num_nodes)
	n, row, col = torch.meshgrid(v2, v1, v1, indexing="ij")
	
	const0 = torch.tensor(0.0, dtype=torch.float32, device=default_device)
	const1 = torch.tensor(1.0, dtype=torch.float32, device=default_device)
	
	v3 = torch.where(row == col, torch.clamp(n * (m / max_num_nodes) - row, const0, const1), const0).type(localTypes.encodeCompType)
	position_encodings = v3[:, None, :, :].expand(max_num_nodes, D, m, m)
	
	return position_encodings

def create_hv( \
	g_id: int, \
	graph: nx.Graph, \
	out_path: Path, \
	position_encodings: torch.Tensor, \
	query_encodings_fn: Callable[[torch.Tensor | None], torch.Tensor], \
	key_encodings_1_fn: Callable[[torch.Tensor | None], torch.Tensor], \
	key_encodings_2_fn: Callable[[torch.Tensor | None], torch.Tensor], \
	value_encodings_fn: Callable[[torch.Tensor | None], torch.Tensor], \
	ctx1: ICheckpointContext
) -> torch.Tensor:
	node_max_id = graph.number_of_nodes()
	
	ctx1.print(f"Graph {g_id} - Start calculating query")
	current_encodings = query_encodings_fn(None)
	query_hv = hv_functions.query_from_encoded(position_encodings[:node_max_id], current_encodings[:node_max_id])
	ctx1.print(f"Graph {g_id} - Finish calculating query")
	
	ctx1.print(f"Graph {g_id} - Start calculating value")
	value_encodings_fn(current_encodings)
	value_hv = hv_functions.value_from_encoded(position_encodings[:node_max_id], current_encodings[:node_max_id])
	ctx1.print(f"Graph {g_id} - Finish calculating value")
	
	ctx1.print(f"Graph {g_id} - Start calculating key")
	edge_dict: dict[int, tuple[int, int]] = dict()
	for u, v in graph.edges:
		id = commutative_cantor_pairing(u, v)
		if id in edge_dict: continue
		edge_dict[id] = (u, v)
	
	edge_indices1: torch.Tensor = torch.empty((len(edge_dict),), dtype=torch.int32, device=default_device)
	edge_indices2: torch.Tensor = torch.empty((len(edge_dict),), dtype=torch.int32, device=default_device)
	for i, (u, v) in enumerate(edge_dict.values()):
		if u > v: (u, v) = (v, u)
		edge_indices1[i] = u
		edge_indices2[i] = v
	
	key_encodings_1_fn(current_encodings)
	edges1: torch.Tensor = torch.gather(current_encodings, 0, edge_indices1[..., None, None, None].expand(-1, D, m, m))
	key_encodings_2_fn(current_encodings)
	edges2: torch.Tensor = torch.gather(current_encodings, 0, edge_indices2[..., None, None, None].expand(-1, D, m, m))
	del current_encodings
	# torch.cuda.empty_cache()
	key_positions: torch.Tensor = torch.gather(position_encodings, 0, edge_indices2[..., None, None, None].expand(-1, D, m, m))
	
	key_hv = hv_functions.key_from_encoded(edges1, edges2, key_positions)
	del edge_indices1
	del edge_indices2
	del edges1
	del edges2
	del key_positions
	# torch.cuda.empty_cache()
	ctx1.print(f"Graph {g_id} - Finish calculating key")
	
	ctx1.print(f"Graph {g_id} - Start calculating result")
	res: torch.Tensor = hv_functions.attention_function(query_hv, key_hv, value_hv)
	del query_hv
	del key_hv
	del value_hv
	ctx1.print(f"Graph {g_id} - Finish calculating result")
	
	return res

def create_and_save_hv( \
	g_id: int, \
	graph: nx.Graph, \
	out_path: Path, \
	position_encodings: torch.Tensor, \
	query_encodings_fn: Callable[[torch.Tensor | None], torch.Tensor], \
	key_encodings_1_fn: Callable[[torch.Tensor | None], torch.Tensor], \
	key_encodings_2_fn: Callable[[torch.Tensor | None], torch.Tensor], \
	value_encodings_fn: Callable[[torch.Tensor | None], torch.Tensor], \
	ctx1: ICheckpointContext, \
	ctx2: ICheckpointContext \
) -> None:
	ctx2.print(f"Graph {g_id} - Start")
	
	res = create_hv( \
		g_id = g_id, \
		graph = graph, \
		out_path = out_path, \
		position_encodings = position_encodings, \
		query_encodings_fn = query_encodings_fn, \
		key_encodings_1_fn = key_encodings_1_fn, \
		key_encodings_2_fn = key_encodings_2_fn, \
		value_encodings_fn = value_encodings_fn, \
		ctx1 = ctx1
	).to(localTypes.hvCompType)
	
	ctx1.print(f"Graph {g_id} - Start saving result")
	torch.save(res, out_path)
	ctx1.print(f"Graph {g_id} - Finish saving result")
	
	ctx2.print(f"Graph {g_id} - Finish")

def action_create_hv(g_id: int, root: FsOrganizer) -> None:
	max_num_nodes = get_dataset_main().max_num_nodes
    
	query_encodings_fn = lambda out: get_complex_random_hvs(D, m, root.query_encodings, \
	    max_num_nodes, device=default_device, out=out)
	key_encodings_1_fn = lambda out: get_complex_random_hvs(D, m, root.key_encodings_1, \
	    max_num_nodes, device=default_device, out=out)
	key_encodings_2_fn = lambda out: get_complex_random_hvs(D, m, root.key_encodings_2, \
	    max_num_nodes, device=default_device, out=out)
	value_encodings_fn = lambda out: get_complex_random_hvs(D, m, root.value_encodings, \
	    max_num_nodes, device=default_device, out=out)
	
	ctx1 = VoidCheckpointContext()
	ctx2 = VoidCheckpointContext()
	
	position_encodings = get_position_encodings()
	
	graphs = get_graph_dataset(root.tudataset)
	graph = next(itertools.islice(graphs, g_id, None))

	create_and_save_hv( \
		g_id = g_id, \
		graph = graph, \
		out_path = root.hv_encoding_of(g_id), \
		position_encodings = position_encodings, \
		query_encodings_fn = query_encodings_fn, \
		key_encodings_1_fn = key_encodings_1_fn, \
		key_encodings_2_fn = key_encodings_2_fn, \
		value_encodings_fn = value_encodings_fn, \
		ctx1 = ctx1, \
		ctx2 = ctx2 \
	)

def action_create_all_hvs(g_id: int, root: FsOrganizer) -> None:
	query_encodings_fn = lambda out: get_complex_random_hvs(D, m, root.query_encodings, m, device=default_device, out=out)
	key_encodings_1_fn = lambda out: get_complex_random_hvs(D, m, root.key_encodings_1, m, device=default_device, out=out)
	key_encodings_2_fn = lambda out: get_complex_random_hvs(D, m, root.key_encodings_2, m, device=default_device, out=out)
	value_encodings_fn = lambda out: get_complex_random_hvs(D, m, root.value_encodings, m, device=default_device, out=out)
	
	ctx1 = CheckpointContext(f"Graph - individual parts")
	ctx2 = CheckpointContext(f"Graph - whole graph")
	
	position_encodings = get_position_encodings()
	
	graphs = get_mutag_dataset(root.tudataset)
	
	for g_id, graph in enumerate(graphs):
		create_and_save_hv( \
			g_id = g_id, \
			graph = graph, \
			out_path = root.hv_encoding_of(g_id), \
			position_encodings = position_encodings, \
			query_encodings_fn = query_encodings_fn, \
			key_encodings_1_fn = key_encodings_1_fn, \
			key_encodings_2_fn = key_encodings_2_fn, \
			value_encodings_fn = value_encodings_fn, \
			ctx1 = ctx1, \
			ctx2 = ctx2 \
		)

__all__ = ["action_create_hv"]

