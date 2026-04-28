from pathlib import Path
from typing import Iterable, Callable

from .model import Model
from tudataset import get_dataset_info
from fs_organization import FsOrganizer
from fn_context import FnContext
from hv_functions import UpperTensorFunctionsManager
from hv_proxy import HVProxy, iter_from_fs as proxies_from_fs
from utils import get_train_and_test_datasets
from distribute_rows import distribute_rows
from test_model import test_model
from time_ import Timer
from process_results import process_results
from get_args import get_arg

def train_model(instance_id: int, id: int, src: int | None, *, msg: str | None = None) -> Callable[[FnContext], None]:
    def inner(ctx: FnContext) -> None:
        fs = ctx.fs
        fs.config.model_dir   = f"instances/{instance_id}/models/{id}"
        fs.config.dist_file   = f"instances/{instance_id}/dist_file.json"
        fs.config.result_file = f"instances/{instance_id}/result_file.json"
        fs.setup()
        
        model: Model
        if src is None:
            classes: Iterable[int] = get_dataset_info().label_num.keys()
            model = Model.new_empty(fs.model, classes)
        else:
            prev_model: Model = Model.from_fs(fs.root / f"instances/{instance_id}/models/{src}")
            model = prev_model.clone(fs.model)
            del prev_model
        
        train_ids, _ = get_train_and_test_datasets(ctx.fs.train_and_test_sets_distribution)
        train_proxies = proxies_from_fs(fs, train_ids)
        
        timer = Timer(f"{msg} -> core function" if msg is not None else "Core function")
        model.train(ctx.functions, train_proxies, msg=msg)
        timer.end()
    
    return inner

max_instances = get_arg("TRAIN_INSTANCES", "int")
max_iterations = get_arg("TRAIN_ITERATIONS", "int")
steps = max_iterations + 2
def get_action(action_id: int) -> tuple[str, Callable[[FnContext], None]] | None:
    if action_id < max_instances * steps:
        instance_id = action_id // steps
        step_id = action_id % steps
        v1 = f"Instance {instance_id+1} of {max_instances}"
        instance_name = f"instances/{instance_id}"
        
        if step_id < 1:
            return (f"{v1} -> distribute rows", distribute_rows(instance_name))
        step_id -= 1
        
        if step_id < max_iterations:
            id = step_id
            src = (id - 1) if (id > 0) else None
            msg = f"train model: {id+1} of {max_iterations} iteration(s)"
            full_msg = f"{v1} -> {msg}"
            return (full_msg, train_model(instance_id, id, src, msg=full_msg))
        step_id -= max_iterations
        
        if step_id < 1:
            return (f"{v1} -> test model", test_model(instance_name, f"models/{max_iterations-1}"))
        step_id -= 1
        
        raise Exception("Unexpected error")
        
    action_id -= max_instances * steps
    
    if action_id < 1:
        return ("Aggregate model prediction results",
            process_results((f"instances/{x}/result_file.json" for x in range(max_instances)), "results.json"))
    action_id -= 1
    
    return None
