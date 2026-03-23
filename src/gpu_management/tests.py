from pathlib import Path
from typing import Callable, Iterable
from dataclasses import dataclass
import shutil
import json

import torch

from .memory import MemoryManager
from .tensor_functions import TensorFunctionsManager
from .data_type import DataType

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

tests: dict[str, Callable[[TestContext], None]] = {
    "addition": addition_test,
    "matrix_mult": matrix_mult_test,
    "softmax": softmax_test,
    "summation": summation_test
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
    
    with TensorFunctionsManager({"complex64": 3000, "float32": 3000}) as fns_manager:
        ctx = TestContext(fns_manager = fns_manager, root = root)
        run_tests(ctx)
        
__all__ = ["all_tests"]

