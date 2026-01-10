import os
import sys
import torch
from pathlib import Path
import pickle
from typing import TypeVar
import torch
import networkx as nx
import itertools
import re

import hv_functions
from hv_memory import get_random_hvs
from utils import find_unique_path, CheckpointContext, get_range_tensor, get_single_tensor, commutative_cantor_pairing
from device import default_device
from mutag import get_mutag_dataset

"""
def record_gpu_management(fn: Callable[[], T], out: Path) -> T:
	torch.cuda.memory._record_memory_history()
	
	res = fn()
	
	snapshot = torch.cuda.memory._snapshot()
	torch.cuda.memory._record_memory_history(enabled=None)
	with out.open("wb") as f:
		pickle.dump(snapshot, f, protocol=4)
	
	return res
"""

D = 10000
m = 28

def create_hv( \
	g_id: int, \
	graph: nx.Graph, \
	out_path: Path, \
	position_encodings: torch.Tensor, \
	query_encodings: torch.Tensor, \
	key_encodings_1: torch.Tensor, \
	key_encodings_2: torch.Tensor, \
	value_encodings: torch.Tensor, \
	ctx1: CheckpointContext, \
	ctx2: CheckpointContext \
) -> None:
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
		
	edge_indices1: torch.Tensor = torch.empty((len(edge_dict),), dtype=torch.int32, device=default_device)
	edge_indices2: torch.Tensor = torch.empty((len(edge_dict),), dtype=torch.int32, device=default_device)
	for i, (u, v) in enumerate(edge_dict.values()):
		if u > v: (u, v) = (v, u)
		edge_indices1[i] = u
		edge_indices2[i] = v
		
	edges1: torch.Tensor = torch.gather(key_encodings_1, 0, edge_indices1[..., None, None, None].expand(-1, D, m, m))
	edges2: torch.Tensor = torch.gather(key_encodings_2, 0, edge_indices2[..., None, None, None].expand(-1, D, m, m))
	key_positions: torch.Tensor = torch.gather(position_encodings, 0, edge_indices2[..., None, None, None].expand(-1, D, m, m))
		
	key_hv = hv_functions.key_from_encoded(edges1, edges2, key_positions)
	ctx1.print(f"Graph {g_id} - Finish calculating key")
		
	ctx1.print(f"Graph {g_id} - Start calculating result")
	res: torch.Tensor = hv_functions.attention_function(query_hv, key_hv, value_hv)
	ctx1.print(f"Graph {g_id} - Finish calculating result")

	ctx1.print(f"Graph {g_id} - Start saving result")
	torch.save(res, out_path)
	ctx1.print(f"Graph {g_id} - Finish saving result")
	
	ctx2.print(f"Graph {g_id} - Finish")

def action_create_hv(g_id: int, root_dir: Path) -> None:
	encodings_dir = root_dir / "encodings"
	encodings_dir.mkdir(parents=True, exist_ok=True)
	base_dir = root_dir / "base"
	base_dir.mkdir(parents=True, exist_ok=True)
	
	query_encodings = get_random_hvs(D, m, base_dir / "query.pt", m, device=default_device)
	key_encodings_1 = get_random_hvs(D, m, base_dir / "key1.pt" , m, device=default_device)
	key_encodings_2 = get_random_hvs(D, m, base_dir / "key2.pt" , m, device=default_device)
	value_encodings = get_random_hvs(D, m, base_dir / "value.pt", m, device=default_device)
	
	v1 = get_range_tensor(m)
	n, row, col = torch.meshgrid(v1, v1, v1, indexing="ij")
	v3 = torch.where((n == row) & (n == col), get_single_tensor(1.0), get_single_tensor(0.0))
	position_encodings = v3[:, None, :, :].expand(m, D, m, m)
	
	ctx1 = CheckpointContext(f"Graph - individual parts")
	ctx2 = CheckpointContext(f"Graph - whole graph")
	
	graphs = get_mutag_dataset(root_dir / "tudataset")
	graph = next(itertools.islice(graphs, g_id, None))

	create_hv( \
		g_id = g_id, \
		graph = graph, \
		out_path = encodings_dir / f"{g_id}.pt", \
		position_encodings = position_encodings, \
		query_encodings = query_encodings, \
		key_encodings_1 = key_encodings_1, \
		key_encodings_2 = key_encodings_2, \
		value_encodings = value_encodings, \
		ctx1 = ctx1, \
		ctx2 = ctx2 \
	)

def action_create_hvs(root_dir: Path) -> None:
	encodings_dir = root_dir / "encodings"
	encodings_dir.mkdir(parents=True, exist_ok=True)
	base_dir = root_dir / "base"
	base_dir.mkdir(parents=True, exist_ok=True)
	
	query_encodings = get_random_hvs(D, m, base_dir / "query.pt", m, device=default_device)
	key_encodings_1 = get_random_hvs(D, m, base_dir / "key1.pt" , m, device=default_device)
	key_encodings_2 = get_random_hvs(D, m, base_dir / "key2.pt" , m, device=default_device)
	value_encodings = get_random_hvs(D, m, base_dir / "value.pt", m, device=default_device)
	
	v1 = get_range_tensor(m)
	n, row, col = torch.meshgrid(v1, v1, v1, indexing="ij")
	v3 = torch.where((n == row) & (n == col), get_single_tensor(1.0), get_single_tensor(0.0))
	position_encodings = v3[:, None, :, :].expand(m, D, m, m)
	
	ctx1 = CheckpointContext(f"Graph - individual parts")
	ctx2 = CheckpointContext(f"Graph - whole graph")
	ctx3 = CheckpointContext(f"Graph - all graphs", msg="Start")
	
	graphs = get_mutag_dataset(root_dir / "tudataset")
	
	for g_id, graph in enumerate(graphs):
		ctx1.print(f"Graph {g_id} - Start emptying GPU cache")
		torch.cuda.empty_cache()
		ctx1.print(f"Graph {g_id} - Finish emptying GPU cache")
		
		create_hv( \
			g_id = g_id, \
			graph = graph, \
			out_path = encodings_dir / f"{g_id}.pt", \
			position_encodings = position_encodings, \
			query_encodings = query_encodings, \
			key_encodings_1 = key_encodings_1, \
			key_encodings_2 = key_encodings_2, \
			value_encodings = value_encodings, \
			ctx1 = ctx1, \
			ctx2 = ctx2 \
		)

	ctx3.print(f"Final push - Start emptying GPU cache")
	torch.cuda.empty_cache()
	ctx3.print(f"Final push - Finish emptying GPU cache")

encode_singular_action_ids_re = re.compile("^encode-(\\d+)$")
def get_action(id: str) -> Callable[[Path], None]:
	if id == "encode":
		return action_create_hvs
	
	match_list = encode_singular_action_ids_re.findall(id)
	if len(match_list) > 0:
		g_id_str = match_list[0]
		g_id = int(g_id_str)
		return lambda path: action_create_hv(g_id, path)
	
	return None

def main() -> None:
	if default_device.type != "cuda":
		print("GPU not available. Exiting")
		return
	
	root_dir_str = os.environ.get("ROOT_DIR", None)
	if root_dir_str is None:
		print("Env var ROOT_DIR not present")
		return
	root_dir = Path(root_dir_str)
	
	action_id = os.environ.get("ACTION_ID", None)
	if action_id is None:
		print("Env var ACTION_ID not present")
		return
	action = get_action(action_id)
	if action is None:
		print("Unknown action id")
		return
	
	mem_history_out_str = os.environ.get("MEM_HISTORY_OUT", None)
	mem_history_out = Path(mem_history_out_str) if mem_history_out_str is not None else None
	
	if mem_history_out is not None:
		torch.cuda.memory._record_memory_history()
	
		action(root_dir)
	
		snapshot = torch.cuda.memory._snapshot()
		torch.cuda.memory._record_memory_history(enabled=None)
		with mem_history_out.open("wb") as f:
			pickle.dump(snapshot, f, protocol=4)
	else:
		action(root_dir)
	

if __name__ == "__main__":
	main()
