import os
import torch
from pathlib import Path
from typing import Callable
import pickle
import re
import json

from device import default_device
from encode_graphs import action_create_hv
from mutag import define_ids_to_labels_mapping
from fs_organization import FsOrganizer
from utils import define_train_and_test_datasets, check_int
from hv_proxy import iter_from_fs as proxies_from_fs, iter_to_batch as proxies_to_batch
from model import Model

def train_model(root: FsOrganizer) -> None:
	train_ids, _ = get_train_and_test_datasets(root.train_and_test_sets_distribution)
	train_proxies = proxies_from_fs(root, train_ids)
	trained_model = Model.train(train_proxies)
	trained_model.to_fs(root.model)

def test_model(root: FsOrganizer) -> None:
	trained_model = Model.from_fs(root.model)
	
	_, test_ids = tuple(get_train_and_test_datasets(root.train_and_test_sets_distribution))
	test_proxies = tuple(proxies_from_fs(root, test_ids))
	test_expected_labels = tuple(proxy.label for proxy in test_proxies)
	
	test_data: torch.Tensor = proxies_to_batch(test_proxies)
	test_result_labels_tensor: torch.Tensor = trained_model.test(test_data)
	test_result_labels: tuple[int, ...] = tuple(check_int(x.item()) for x in test_result_labels_tensor)
	
	json_data = json.dumps({"ids": test_ids, "expected": test_expected_labels, "result": test_result_labels})
	root.test_results.write_text(json_data)

encode_singular_action_ids_re = re.compile("^encode-(\\d+)$")
def get_action(id_str: str) -> Callable[[FsOrganizer], None] | None:
	id: int
	try:
		id = int(id_str)
	except ValueError:
		return None
	
	if id < 0:
		return None
	elif id == 0:
		return lambda root: root.setup()
	elif id < 189:
		g_id = id - 1
		return lambda root: action_create_hv(g_id, root)
	elif id == 189:
		return lambda root: define_ids_to_labels_mapping(root.ids_to_labels)
	elif id == 190:
		return lambda root: define_train_and_test_datasets(root.train_and_test_sets_distribution)
	elif id == 191:
		return train_model
	elif id == 192:
		return test_model
	else:
		return None

def main() -> None:
	if default_device.type != "cuda":
		print("GPU not available. Exiting")
		return
	
	root_dir_str = os.environ.get("ROOT_DIR", None)
	if root_dir_str is None:
		raise Exception("Env var ROOT_DIR not present")
	root_dir = Path(root_dir_str)
	
	action_id = os.environ.get("ACTION_ID", None)
	if action_id is None:
		raise Exception("Env var ACTION_ID not present")
	action = get_action(action_id)
	if action is None:
		raise Exception("Unknown action id")
	mem_history_out_str = os.environ.get("MEM_HISTORY_OUT", None)
	mem_history_out = Path(mem_history_out_str) if mem_history_out_str is not None else None
	
	fs_organizer = FsOrganizer(root_dir)
	
	if mem_history_out is not None:
		mem_history_out.parent.mkdir(parents=True, exist_ok=True)
		
		torch.cuda.memory._record_memory_history()
	
		action(fs_organizer)
	
		snapshot = torch.cuda.memory._snapshot()
		torch.cuda.memory._record_memory_history(enabled=None)
		with mem_history_out.open("wb") as f:
			pickle.dump(snapshot, f, protocol=4)
	else:
		action(fs_organizer)


if __name__ == "__main__":
	main()
