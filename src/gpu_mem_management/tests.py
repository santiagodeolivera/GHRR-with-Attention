from pathlib import Path
from typing import Callable, Iterable
from dataclasses import dataclass
import shutil
import json

import torch

from memory import MemoryManager
from tensor_functions import TensorFunctionsManager
from constants import DataType
from operation_manager import OperationManager

@dataclass
class TestContext:
    fns_manager: TensorFunctionsManager
    root: Path

def addition_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = ctx.root
    
    p1 = fns.randn((10, 10, 10, 10, 10), root / "a1", DataType.complex64)
    p2 = fns.randn((10, 10, 10, 10, 10), root / "a2", DataType.complex64)
    p3 = fns.addition(p1, p2, out=root / "a3")
    
    t1 = p1.tensor()
    t2 = p2.tensor()
    t3 = p3.tensor()
    
    result = t1 + t2
    if not torch.allclose(t3, result):
        raise Exception()

def matrix_mult_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = ctx.root
    
    p1 = fns.randn((5, 4, 3, 10, 2), root / "mm1", DataType.complex64)
    p2 = fns.randn((5, 4, 3, 2, 4), root / "mm2", DataType.complex64)
    p3 = fns.matrix_mult(p1, p2, out=root / "mm3")
    
    t1 = p1.tensor()
    t2 = p2.tensor()
    t3 = p3.tensor()
    
    result = t1 @ t2
    if not torch.allclose(t3, result):
        raise Exception()

def softmax_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = ctx.root
    
    p1 = fns.randn((5, 4, 2, 3, 10), root / "softmax1", DataType.float32)
    p2 = fns.softmax(p1, root / "softmax2")
    
    t1 = p1.tensor()
    t2 = p2.tensor()
    
    result = torch.nn.functional.softmax(t1, dim=-1)
    if not torch.allclose(t2, result):
        raise Exception()

def summation_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = ctx.root
    
    p1 = fns.randn((5, 4, 2, 3, 10), root / "summation", DataType.complex64)
    p2 = fns.summation(p1, 1, root / "summation2")
    p3 = fns.summation(p1, 2, root / "summation3")
    
    t1 = p1.tensor()
    t2 = p2.tensor()
    t3 = p3.tensor()
    
    result2 = t1.sum(dim=-1)
    result3 = t1.sum(dim=(-1, -2))
    
    if not torch.allclose(t2, result2):
        raise Exception()
    
    if not torch.allclose(t3, result3):
        raise Exception()

def operation_manager_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = ctx.root
    
    op_root = root / "op_manager"
    op_manager = OperationManager(op_root, fns)
    
    templates: Iterable[tuple[tuple[int, ...], str, DataType]] = (
        ((10, 10, 10), "1", DataType.complex64),
        ((10, 5, 2, 4), "2", DataType.float32)
    )
    with open(root / "op_manager/operations.csv", "a") as file:
        for shape, name, dtype in templates:
            line = f"{name} ; randn ; {json.dumps(shape)} ; {dtype.name}"
            file.write(line)
            file.write("\n")
    
    op_manager.execute_all()
    
    for shape, name, dtype in templates:
        proxy = op_manager.get_tensor(name)
        tensor = proxy.tensor()
        
        if tensor.shape != shape:
            raise Exception()
        
        if tensor.dtype != dtype.value:
            raise Exception()

tests: dict[str, Callable[[TestContext], None]] = {
    "addition": addition_test,
    "matrix_mult": matrix_mult_test,
    "softmax": softmax_test,
    "summation": summation_test,
    "operation_manager": operation_manager_test
}

def run_tests(ctx: TestContext) -> None:
    for k, v in tests.items():
        print(f"Starting test {repr(k)}")
        v(ctx)
        print(f"Test {repr(k)} successful")

def all_tests() -> None:
    root = Path(__file__).resolve().parent.parent.parent.parent / "test/outputs"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    
    with MemoryManager.create(3000, data_type=DataType.complex64) as memory_manager, \
        MemoryManager.create(3000, data_type=DataType.float32) as real_n_memory_manager:
        
        fns_manager = TensorFunctionsManager(memory_manager, real_n_memory_manager)
        ctx = TestContext(fns_manager = fns_manager, root = root)
        run_tests(ctx)
        
__all__ = ["all_tests"]

