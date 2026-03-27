from pathlib import Path
from typing import Callable
from dataclasses import dataclass
import shutil

import torch

from .tensor_functions import TensorFunctionsManager
from .data_type import DataType
from mmap_tensors import MmapTensors

@dataclass
class TestContext:
    fns_manager: TensorFunctionsManager
    root: Path

def addition_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = ctx.root
    
    shape = (10,) * 5
    t1, t2, t3 = tuple(MmapTensors.new_override(root / f"a{n}", shape, DataType.complex64) for n in (1, 2, 3))
    fns.randn(shape, DataType.complex64, out=t1)
    fns.randn(shape, DataType.complex64, out=t2)
    fns.addition(t1, t2, out=t3)
    
    result = t1 + t2
    if not torch.allclose(t3, result):
        raise Exception()

def matrix_mult_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = ctx.root
    
    t1, t2, t3 = tuple(MmapTensors.new_override(root / f"mm{n}", shape, DataType.complex64) for n, shape in (
        (1, (5, 4, 3, 10, 2)),
        (2, (5, 4, 3, 2, 4)),
        (3, (5, 4, 3, 10, 4))
    ))
    fns.randn((5, 4, 3, 10, 2), DataType.complex64, out=t1)
    fns.randn((5, 4, 3, 2, 4), DataType.complex64, out=t2)
    fns.matrix_mult(t1, t2, out=t3)
    
    result = t1 @ t2
    if not torch.allclose(t3, result):
        raise Exception()

def softmax_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = ctx.root
    
    t1, t2 = tuple(MmapTensors.new_override(root / f"softmax{n}", (5, 4, 2, 3, 10), DataType.float32) for n in (1, 2))
    fns.randn((5, 4, 2, 3, 10), DataType.float32, out=t1)
    fns.softmax(t1, out=t2)
    
    result = torch.nn.functional.softmax(t1, dim=-1)
    if not torch.allclose(t2, result):
        raise Exception()

def summation_test(ctx: TestContext) -> None:
    fns = ctx.fns_manager
    root = ctx.root
    
    t1, t2, t3, t4, t5 = tuple(MmapTensors.new_override(root / f"sum{n}", shape, DataType.complex64) for n, shape in (
        (1, (5, 4, 2, 3, 10)),
        (2, (5, 4, 2, 3)),
        (3, (5, 4, 2)),
        (4, (5, 2, 3, 10)),
        (5, (5, 2, 4, 10))
    ))
    
    fns.randn((5, 4, 2, 3, 10), DataType.complex64, out=t1)
    fns.summation(t1, 1, out=t2)
    fns.summation(t1, 2, out=t3)
    fns.summation(t1.transpose(-4, -3).transpose(-3, -2).transpose(-2, -1), 1, out=t4)
    fns.summation(t1.transpose(-4, -3).transpose(-2, -1), 1, out=t5)
    
    result2 = t1.sum(dim=-1)
    result3 = t1.sum(dim=(-1, -2))
    result4 = t1.sum(dim=-4)
    result5 = t1.transpose(-4, -3).sum(dim=-2)
    
    if not torch.allclose(t2, result2):
        raise Exception()
    
    if not torch.allclose(t3, result3):
        raise Exception()
    
    if not torch.allclose(t4, result4):
        raise Exception()
    
    if not torch.allclose(t5, result5):
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

