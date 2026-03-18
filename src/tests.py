from pathlib import Path
from contextlib import ExitStack
from typing import Callable

import torch

import tensor_functions as F

def addition_test() -> None:
    root = Path(__file__).resolve().parent.parent / "test_outputs"
    
    p1 = F.randn((10, 10, 10, 10, 10), root / "1")
    p2 = F.randn((10, 10, 10, 10, 10), root / "2")
    p3 = F.addition(p1, p2, out=root / "3")
    
    t1 = p1.tensor()
    t2 = p2.tensor()
    t3 = p3.tensor()
    
    result = t1 + t2
    if not torch.allclose(t3, result):
        raise Exception()

tests: dict[str, Callable[[], None]] = {
    "addition": addition_test
}

def all_tests() -> None:
    for k, v in tests.items():
        print(f"Starting test {repr(k)}")
        v()
        print(f"Test {repr(k)} successful")

__all__ = ["all_tests"]

