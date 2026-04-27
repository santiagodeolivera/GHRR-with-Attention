from typing import Callable

from fn_context import FnContext
from utils import define_train_and_test_datasets

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

