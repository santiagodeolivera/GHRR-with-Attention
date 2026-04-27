import json
from typing import Callable

import torch

from fn_context import FnContext
from utils import get_train_and_test_datasets, check_int
from time_ import Timer
from hv_proxy import iter_to_batch as proxies_to_batch, iter_from_fs as proxies_from_fs
from model import Model

def test_model(instance_name: str, model_path: str = "model") -> Callable[[FnContext], None]:
    def inner(ctx: FnContext) -> None:
        root = ctx.fs
        functions = ctx.functions
        
        root.config.model_dir = f"{instance_name}/{model_path}"
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
        
        test_result_labels_mid = torch.empty(len(test_proxies), dtype=torch.int8)
        mid_start = 0
        step = 100
        total_end = len(test_proxies)
        while mid_start < total_end:
            mid_end = min(mid_start + step, total_end)
            mid_test_proxies = test_proxies[mid_start:mid_end]
            timer0 = Timer(f"Handle test HVs: [{mid_start}; {mid_end}) of [0; {total_end})")
            
            timer = Timer("Create HV batch")
            mid_test_data: torch.Tensor = proxies_to_batch(mid_test_proxies)
            del mid_test_proxies
            timer.end()
            
            timer = Timer("Predict test dataset graphs")
            mid_test_result_labels_tensor: torch.Tensor = trained_model.test(functions, mid_test_data)
            del mid_test_data
            timer.end()
            
            test_result_labels_mid[mid_start:mid_end] = mid_test_result_labels_tensor
            del mid_test_result_labels_tensor
            
            timer0.end()
            mid_start = mid_end
        
        test_result_labels: tuple[int, ...] = tuple(check_int(x.item()) for x in test_result_labels_mid)
        
        with open(root.result_file, "w") as file:
            json.dump({"ids": test_ids, "expected": test_expected_labels, "result": test_result_labels}, file)
    
    return inner

