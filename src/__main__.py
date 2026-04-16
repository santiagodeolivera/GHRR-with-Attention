import prev_setup

from time_ import Timer

setup_timer = Timer("Initial setup")

import time
from typing import Callable
import re
import json
import abc

import torch

from encode_graphs import action_create_hv
from tudataset import define_ids_to_labels_mapping
from fs_organization import FsOrganizer
from hv_proxy import iter_from_fs as proxies_from_fs, iter_to_batch as proxies_to_batch
from model import Model
from process_results import process_results
from f2 import func as f2_function
from f3 import get_action as get_f3_action
from fn_context import FnContext
from gpu_management.tensor_functions import TensorFunctionsManager
from gpu_management.tests import all_tests as gpu_tests
from hv_functions import UpperTensorFunctionsManager
from get_args import get_arg
from utils import define_train_and_test_datasets, get_train_and_test_datasets, check_int
from time_ import time_start
from tudataset import get_dataset_info

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
        
        timer = Timer("Get model from FS")
        trained_model = Model.from_fs(root.model)
        timer.end()
        
        timer = Timer("Get proxies and expected labels")
        _, test_ids = tuple(get_train_and_test_datasets(root.train_and_test_sets_distribution))
        test_proxies = tuple(proxies_from_fs(root, test_ids))
        test_expected_labels = tuple(proxy.label for proxy in test_proxies)
        timer.end()
        
        timer = Timer("Create HV batch")
        test_data: torch.Tensor = proxies_to_batch(test_proxies)
        timer.end()
        
        timer = Timer("Predict test dataset graphs")
        test_result_labels_tensor: torch.Tensor = trained_model.test(functions, test_data)
        timer.end()
        
        test_result_labels: tuple[int, ...] = tuple(check_int(x.item()) for x in test_result_labels_tensor)
    
        with open(root.result_file, "w") as file:
            json.dump({"ids": test_ids, "expected": test_expected_labels, "result": test_result_labels}, file)
    
    return inner

encode_singular_action_ids_re = re.compile("^encode-(\\d+)$")
def get_action(program_id: int, action_id: int) -> tuple[str, Callable[[FnContext], None]] | None:
    if action_id < 0:
        raise Exception(f"Invalid action_id: {action_id}")
    
    if program_id == 1:
        num_graphs = get_dataset_info().num_graphs
        num_models = 10
        
        if action_id < 1:
            return ("Setup", lambda ctx: ctx.fs.setup())
        action_id -= 1
        
        if action_id < num_graphs:
            g_id = action_id
            return (f"Create HV {g_id+1} of {num_graphs}", lambda root: action_create_hv(g_id, root))
        action_id -= num_graphs
        
        if action_id < 1:
            return ("Define ids to labels mapping", lambda ctx: define_ids_to_labels_mapping(ctx.fs.tudataset, ctx.fs.ids_to_labels))
        action_id -= 1
        
        if action_id < num_models * 3:
            counter = action_id
            instance_id = counter // 3
            instance_step = counter % 3
            
            instance_dir = f"instances/{instance_id}"
            
            v1 = f"Model {instance_id+1} of {num_models}"
            if instance_step == 0:
                return (f"{v1} -> distribute rows", distribute_rows(instance_dir))
            elif instance_step == 1:
                return (f"{v1} -> train model", train_model(instance_dir))
            elif instance_step == 2:
                return (f"{v1} -> test model", test_model(instance_dir))
        action_id -= num_models * 3
        
        if action_id < 1:
            return ("Aggregate model prediction results",
                process_results((f"instances/{x}/result_file.json" for x in range(10)), "results.json"))
        action_id -= 1
        
        return None
    elif program_id == 2:
        if action_id < 1:
            return ("Compare several random HVs of all classes", f2_function)
        action_id -= 1
        return None
    elif program_id == 3:
        return get_f3_action(action_id)
    else:
        raise ValueError("Unknown program id")
    
    return None

class ActionLimiter(abc.ABC):
    __current: int
    
    def __init__(self, start: int) -> None:
        self.__current = start
    
    @abc.abstractmethod
    def to_next(self) -> bool: ...
    
    def next(self) -> int | None:
        if not self.to_next():
            return None
        
        result = self.__current
        self.__current += 1
        return result

class RangeLimiter(ActionLimiter):
    __end: int
    
    def __init__(self, start: int) -> None:
        super().__init__(start)
        self.__end = get_arg("END", "int")
    
    def to_next(self) -> bool:
        return self.__current <= self.__end

class TimeLimiter(ActionLimiter):
    __stop_timestamp: int
    
    def __init__(self, start: int) -> None:
        super().__init__(start)
        self.__stop_timestamp = time_start + int( get_arg("MINUTES", "float") * 60 * 1_000_000_000 )
    
    def to_next(self) -> bool:
        return time.time_ns() < self.__stop_timestamp

limiters: dict[str, Callable[[int], ActionLimiter]] = {
    "RANGE": RangeLimiter,
    "TIMED": TimeLimiter
}
def default_program(mode: str) -> None:
    vars_timer = Timer("Secondary setup")
    
    if not torch.cuda.is_available():
        raise Exception("CUDA not available")
    
    root_dir = get_arg("ROOT_DIR", "Path")
    program_id = get_arg("PROGRAM_ID", "int")
    max_gpu_mem = get_arg("MAX_GPU_MEM", "int")
    
    start = get_arg("START", "int")
    end: int | None = None
    
    limiter_fn = limiters.get(mode, None)
    if limiter_fn is None:
        raise Exception(f"Unknown limiter mode: {mode}")
    limiter = limiter_fn(start)
    
    with TensorFunctionsManager(max_gpu_mem) as lower_manager:
        upper_manager = UpperTensorFunctionsManager(lower_manager)
        vars_timer.end()
        
        try:
            while True:
                action_id = limiter.next()
                if action_id is None:
                    break
                
                action_data = get_action(program_id, action_id)
                if action_data is None:
                    break
                action_name, action = action_data
                
                ctx = FnContext(fs = FsOrganizer(root_dir), functions = upper_manager)
                timer = Timer(f"Program {program_id}, action {action_id}: {action_name}")
                
                try:
                    action(ctx)
                except BaseException as e:
                    timer.error()
                    raise e
                
                timer.end()
                end = action_id
        finally:
            if end is not None:
                print(f"Executed program {program_id}, steps {start} to {end}")
            else:
                print(f"No steps were executed for program {program_id}")


def main() -> None:
    mode = get_arg("MODE", "str")
    match mode:
        case "RANGE" | "TIMED":
            default_program(mode)
        case "INFO":
            info = get_dataset_info()
            print("Source dataset:", info.name)
            print("Number of graphs:", info.num_graphs)
            print("Max number of nodes in a graph:", info.max_num_nodes)
            print("Max number of edges in a graph:", info.max_num_edges)
            for label, num in info.label_num.items():
                print(f"{num} graphs of label {repr(label)}")
        case "TEST":
            gpu_tests()
        case _:
            raise Exception(f"Unknown mode: {repr(mode)}")

setup_timer.end()

if __name__ == "__main__":
    main()

