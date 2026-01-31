import os
import torch
from pathlib import Path
from typing import Callable
import pickle
import re

from device import default_device
from encode_graphs import action_create_hv

encode_singular_action_ids_re = re.compile("^encode-(\\d+)$")
def get_action(id_str: str) -> Callable[[Path], None] | None:
	id: int
	try:
		id = int(id_str)
	except ValueError:
		return None
	
	if id < 0:
		return None
	elif id < 188:
		g_id = id
		return lambda path: action_create_hv(g_id, path)
	else:
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
		mem_history_out.parent.mkdir(parents=True, exist_ok=True)
		
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
