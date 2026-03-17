from pathlib import Path
from contextlib import ExitStack

import torch

from memory import MemoryManager

def memory_test() -> None:
    root = Path(__file__).resolve().parent.parent
    print("Defining manager")
    shape = (10, 10, 10)
    
    with MemoryManager.get() as manager:
        with ExitStack() as stack:
            print("Allocating memory")
            t1, t2, t3 = tuple(
                stack.enter_context(
                    manager.empty(shape)
                ) for _ in range(3)
            )
            
            print("Creating random tensors")
            torch.randn(*shape, out=t1.tensor)
            t1.to_fs(root / "test_outputs/t1.pt")
            
            torch.randn(*shape, out=t2.tensor)
            t2.to_fs(root / "test_outputs/t2.pt")
            
            print("Adding tensors")
            torch.add(t1.tensor, t2.tensor, out=t3.tensor)
            t3.to_fs(root / "test_outputs/t3.pt")

        with ExitStack() as stack:
            print("Loading tensors")
            t1, t2, t3 = tuple(
                stack.enter_context(
                    manager.load(root / f"test_outputs/t{n}.pt")
                ) for n in range(1, 4)
            )
            
            print("Adding tensors")
            torch.add(t1.tensor, t2.tensor, out=t1.tensor)
            
            print("Evaluating result")
            if not torch.equal(t1.tensor, t3.tensor):
                raise Exception("Addition check failed")

tests = {
    "memory": memory_test
}

def all_tests() -> None:
    for k, v in tests.items():
        print(f"Starting test {repr(k)}")
        v()
        print(f"Test {repr(k)} successful")

__all__ = ["all_tests"]

