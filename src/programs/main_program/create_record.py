from pathlib import Path
from operation_manager import OperationManagerRecord

from .tudataset import get_dataset_main
from .operation_templates import OperationTemplate, BranchTemplate, random_hvs, query_from_encoded, key_from_encoded, value_from_encoded, templates_to_file, slicerange

def get_operations() -> OperationTemplate:
    info = get_dataset_main()
    max_nodes = info.max_nodes
    
    base_q = "base/q"
    base_k1 = "base/k1"
    base_k2 = "base/k2"
    base_v = "base/v"
    return BranchTemplate(
        *(random_hvs(name, max_nodes) for name in (base_q, base_k1, base_k2, base_v)),
        
    )

def func(root: Path) -> None:
    op = OperationManagerRecord(root)
    op.setup()
    
    
