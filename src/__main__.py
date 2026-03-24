import prev_setup

import os
import torch
from pathlib import Path
from typing import Callable
import pickle
import re
import json

from encode_graphs import action_create_hv
from tudataset import define_ids_to_labels_mapping
from fs_organization import FsOrganizer
from utils import define_train_and_test_datasets, get_train_and_test_datasets, check_int, Timer
from hv_proxy import iter_from_fs as proxies_from_fs, iter_to_batch as proxies_to_batch
from model import Model
from process_results import process_results
from f2 import func as f2_function
from f3 import get_action as get_f3_action
from fn_context import FnContext
from gpu_management import TensorFunctionsManager, TensorProxy
from hv_functions import UpperTensorFunctionsManager

def distribute_rows(instance_name: str) -> Callable[[FnContext], None]:
    def inner(ctx: FnContext) -> None:
        root = ctx.fs
        functions = ctx.functions
        
        root.config.model_dir = f"{instance_name}/model"
        root.config.dist_file = f"{instance_name}/dist_file.json"
        root.config.result_file = f"{instance_name}/result_file.json"
        
        root.setup()
        
        define_train_and_test_datasets(root.train_and_test_sets_distribution)
    
    return inner

def train_model(instance_name: str) -> Callable[[FnContext], None]:
    def inner(ctx: FnContext) -> None:
        root = ctx.fs
        functions = ctx.functions
        
        root.config.model_dir = f"{instance_name}/model"
        root.config.dist_file = f"{instance_name}/dist_file.json"
        root.config.result_file = f"{instance_name}/result_file.json"
        
        train_ids, _ = get_train_and_test_datasets(root.train_and_test_sets_distribution)
        train_proxies = proxies_from_fs(root, train_ids)
        trained_model = Model.train(functions, train_proxies, root.model)
    
    return inner
    
def test_model(instance_name: str) -> Callable[[FnContext], None]:
    def inner(ctx: FnContext) -> None:
        root = ctx.fs
        functions = ctx.functions
        
        root.config.model_dir = f"{instance_name}/model"
        root.config.dist_file = f"{instance_name}/dist_file.json"
        root.config.result_file = f"{instance_name}/result_file.json"
        
        trained_model = Model.from_fs(root.model)
        
        _, test_ids = tuple(get_train_and_test_datasets(root.train_and_test_sets_distribution))
        test_proxies = tuple(proxies_from_fs(root, test_ids))
        test_expected_labels = tuple(proxy.label for proxy in test_proxies)
        
        (tmp,) = functions.tmp_gen.new_paths(1)
        test_data: TensorProxy = proxies_to_batch(test_proxies, out=tmp)
        test_result_labels_tensor: torch.Tensor = trained_model.test(functions, test_data.tensor())
        test_result_labels: tuple[int, ...] = tuple(check_int(x.item()) for x in test_result_labels_tensor)
    
        json_data = json.dumps({"ids": test_ids, "expected": test_expected_labels, "result": test_result_labels})
        root.result_file.write_text(json_data)
    
    return inner

encode_singular_action_ids_re = re.compile("^encode-(\\d+)$")
def get_action(program_id: int, action_id: int) -> Callable[[FnContext], None] | None:
    if program_id == 1:
        if action_id < 0:
            raise ValueError("Out of bounds")
        elif action_id == 0:
            return lambda ctx: ctx.fs.setup()
        elif action_id < 189:
            g_id = action_id - 1
            return lambda root: action_create_hv(g_id, root)
        elif action_id == 189:
            return lambda ctx: define_ids_to_labels_mapping(ctx.fs.tudataset, ctx.fs.ids_to_labels)
        elif action_id < 220:
            counter = action_id - 190
            instance_id = counter // 3
            instance_step = counter % 3
            
            instance_dir = f"instances/{instance_id}"
            
            if instance_step == 0:
                return distribute_rows(instance_dir)
            elif instance_step == 1:
                return train_model(instance_dir)
            elif instance_step == 2:
                return test_model(instance_dir)
        elif action_id == 220:
            return process_results((f"instances/{x}/result_file.json" for x in range(10)), "results.json")
        else:
            raise ValueError("Out of bounds")
    elif program_id == 2:
        if action_id < 0:
            raise ValueError("Out of bounds")
        elif action_id < 190:
            return get_action(1, action_id)
        elif action_id == 190:
            return f2_function
        else:
            raise ValueError("Out of bounds")
    elif program_id == 3:
        if action_id < 0:
            raise ValueError("Out of bounds")
        elif action_id < 221:
            return get_action(1, action_id)
        else:
            return get_f3_action(action_id - 221)
    else:
        raise ValueError("Unknown program id")
    
    return None

def main() -> None:
    if not torch.cuda.is_available():
        raise Exception("CUDA not available")
    
    root_dir_str = os.environ.get("ROOT_DIR", None)
    if root_dir_str is None:
        raise Exception("Env var ROOT_DIR not present")
    root_dir = Path(root_dir_str)
    
    program_id_str = os.environ.get("PROGRAM_ID", None)
    if program_id_str is None:
        raise Exception("Env var ROOT_DIR not present")
    program_id = int(program_id_str)

    action_id_str = os.environ.get("ACTION_ID", None)
    if action_id_str is None:
        raise Exception("Env var ACTION_ID not present")
    action_id = int(action_id_str)
    
    action = get_action(program_id, action_id)
    if action is None:
        raise Exception("Unknown action id")
    
    fs_organizer = FsOrganizer(root_dir)
    
    mem_manager_sizes: dict[str, int] = {
        "complex64": 1024 * 1024 * 1024 * 3 // 8,
        "float32":   1024 * 1024 * 1024 * 3 // 4
    }
    with TensorFunctionsManager(mem_manager_sizes) as lower_manager:
        upper_manager = UpperTensorFunctionsManager(lower_manager, lambda n: fs_organizer.root / f"tmp/{n}")
        ctx = FnContext(fs = fs_organizer, functions = upper_manager)
        
        timer = Timer()
        action(ctx)
        timer.msg("Action finished")

if __name__ == "__main__":
    main()

