from dataclasses import dataclass
from functools import reduce
import os
from pathlib import Path
from contextlib import ExitStack
from typing import Iterable

import torch

element_type = torch.complex64

@dataclass
class TensorProxy:
    __start: int
    __length: int
    __parent: "MemoryManager"
    __tensor: torch.Tensor | None
    
    def __init__(self, start: int, length: int, parent: "MemoryManager", tensor: torch.Tensor) -> None:
        self.__start = start
        self.__length = length
        self.__parent = parent
        self.__tensor = tensor
    
    @property
    def start(self) -> int:
        return self.__start
    
    @property
    def length(self) -> int:
        return self.__length
    
    @property
    def end(self) -> int:
        return self.__start + self.__length
    
    @property
    def tensor(self) -> torch.Tensor:
        if self.__tensor is None:
            raise Exception("Trying to get tensor on a freed proxy")
        
        return self.__tensor
    
    def to_fs(self, path: str | os.PathLike) -> None:
        torch.save(self.__tensor, path)
    
    def load(self, path: str | os.PathLike) -> None:
        cpu_tensor = torch.load(path, map_location="cpu")
        if cpu_tensor.shape != self.tensor.shape:
            raise Exception(f"Attempted to load a {cpu_tensor.shape} file tensor into a {self.tensor.shape} GPU tensor.")
        self.tensor[...] = cpu_tensor
    
    def __enter__(self) -> "TensorProxy":
        return self
    
    def __exit__(self, exc_t, exc_v, exc_tb) -> None:
        self.__parent._remove_proxy(self)
        self.__tensor = None
        self.__start = 0
        self.__length = 0

class MemoryManager:
    # Records must always be ordered by start index
    __records: list[TensorProxy]
    __max_mem: int
    __tensor: torch.Tensor
    
    def __init__(self, tensor: torch.Tensor, max_mem: int) -> None:
        self.__records = []
        self.__max_mem = max_mem
        self.__tensor = tensor
    
    @property
    def max_mem(self) -> int:
        return self.__max_mem
    
    @staticmethod
    def create(max_mem_options: int | Iterable[int]) -> "MemoryManager":
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available")
        
        torch.set_default_device("cuda:0")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        max_mem_allowed = total_memory // 8 // 2
        
        max_mem: int | None = None
        if type(max_mem_options) == int:
            if max_mem_options <= max_mem_allowed:
                max_mem = max_mem_options
        else:
            for option in max_mem_options:
                if option <= max_mem_allowed:
                    max_mem = option
                    break
        
        if max_mem is None:
            raise Exception(f"Not enough GPU memory.")
        
        tensor = torch.empty(max_mem, dtype=element_type)
        
        result = MemoryManager(tensor, max_mem)
        return result
    
    def empty(self, shape: tuple[int, ...]) -> TensorProxy:
        length = reduce(lambda a, b: a * b, shape, 1)
        rec_len = len(self.__records)
        
        for i in range(-1, rec_len):
            min_pos = self.__records[i].end if i >= 0 else 0
            max_pos = self.__records[i + 1].start if i + 1 < rec_len else self.__max_mem
            if length <= max_pos - min_pos:
                new_record = TensorProxy(min_pos, length, self, self.__tensor[min_pos : min_pos + length].view(*shape))
                self.__records.insert(i + 1, new_record)
                return new_record
        
        raise Exception(f"Not enough memory for new tensor of shape {shape}")
    
    def load(self, path: str | os.PathLike) -> TensorProxy:
        cpu_tensor = torch.load(path, map_location="cpu")
        result = self.empty(cpu_tensor.shape)
        result.tensor[...] = cpu_tensor
        return result
    
    def _remove_proxy(self, proxy: TensorProxy) -> None:
        for i in range(len(self.__records)):
            if id(i) == id(proxy):
                del self.__records[i]
                return
    
    def __del__(self) -> None:
        del self.__tensor
        torch.cuda.empty_cache()

def test() -> None:
    root = Path(__file__).resolve().parent.parent
    print("Defining manager")
    manager = MemoryManager.create(3000)
    
    shape = (10, 10, 10)
    
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

__all__ = ["MemoryManager", "test"]

