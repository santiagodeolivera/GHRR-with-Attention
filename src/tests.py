from pathlib import Path
from contextlib import ExitStack
from typing import Callable
from dataclasses import dataclass

import torch

from memory import MemoryManager
from tensor_functions import TensorFunctionsManager

@dataclass
class TestContext:
    fns_manager: TensorFunctionsManager

def addition_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = Path(__file__).resolve().parent.parent / "test_outputs"
    
    p1 = fns.randn((10, 10, 10, 10, 10), root / "a1")
    p2 = fns.randn((10, 10, 10, 10, 10), root / "a2")
    p3 = fns.addition(p1, p2, out=root / "a3")
    
    t1 = p1.tensor()
    t2 = p2.tensor()
    t3 = p3.tensor()
    
    result = t1 + t2
    if not torch.allclose(t3, result):
        raise Exception()

def matrix_mult_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = Path(__file__).resolve().parent.parent / "test_outputs"
    
    p1 = fns.randn((5, 4, 3, 10, 2), root / "mm1")
    p2 = fns.randn((5, 4, 3, 2, 4), root / "mm2")
    p3 = fns.matrix_mult(p1, p2, out=root / "mm3")
    
    t1 = p1.tensor()
    t2 = p2.tensor()
    t3 = p3.tensor()
    
    result = t1 @ t2
    if not torch.allclose(t3, result):
        raise Exception()

tests: dict[str, Callable[[TestContext], None]] = {
    "addition": addition_test,
    "matrix_mult": matrix_mult_test
}

def run_tests(ctx: TestContext) -> None:
    for k, v in tests.items():
        print(f"Starting test {repr(k)}")
        v(ctx)
        print(f"Test {repr(k)} successful")

def all_tests() -> None:
    with MemoryManager.create(3000) as memory_manager:
        fns_manager = TensorFunctionsManager(memory_manager)
        ctx = TestContext(fns_manager = fns_manager)
        run_tests(ctx)
        
__all__ = ["all_tests"]

