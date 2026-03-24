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
    
    p1 = fns.randn((10, 10, 10, 10, 10), DataType.complex64, out=root / "a1")
    t1 = p1.tensor()
    p2 = fns.randn((10, 10, 10, 10, 10), DataType.complex64, out=root / "a2")
    t2 = p2.tensor()
    p3 = fns.addition(t1, t2, out=root / "a3")
    t3 = p3.tensor()
    
    result = t1 + t2
    if not torch.allclose(t3, result):
        raise Exception()

def matrix_mult_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = ctx.root
    
    p1 = fns.randn((5, 4, 3, 10, 2), DataType.complex64, out=root / "mm1")
    t1 = p1.tensor()
    p2 = fns.randn((5, 4, 3, 2, 4), DataType.complex64, out=root / "mm2")
    t2 = p2.tensor()
    p3 = fns.matrix_mult(t1, t2, out=root / "mm3")
    t3 = p3.tensor()
    
    result = t1 @ t2
    if not torch.allclose(t3, result):
        raise Exception()

def softmax_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = ctx.root
    
    p1 = fns.randn((5, 4, 2, 3, 10), DataType.float32, out=root / "softmax1")
    t1 = p1.tensor()
    p2 = fns.softmax(t1, out=root / "softmax2")
    t2 = p2.tensor()
    
    result = torch.nn.functional.softmax(t1, dim=-1)
    if not torch.allclose(t2, result):
        raise Exception()

def summation_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = ctx.root
    
    p1 = fns.randn((5, 4, 2, 3, 10), DataType.complex64, out=root / "summation")
    t1 = p1.tensor()
    p2 = fns.summation(t1, 1, out=root / "summation2")
    t2 = p2.tensor()
    p3 = fns.summation(t1, 2, out=root / "summation3")
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
    
    with TensorFunctionsManager(3000 * DataType.complex64.size) as fns_manager:
        ctx = TestContext(fns_manager = fns_manager, root = root)
        run_tests(ctx)
        
__all__ = ["all_tests"]

