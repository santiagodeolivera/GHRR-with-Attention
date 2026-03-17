from dataclasses import dataclass
from functools import reduce
import os
from pathlib import Path
from typing import Iterable

import torch

from constants import SliceInfo, element_type

def get_file_tensor(path: str | os.PathLike, slice_info: SliceInfo | None) -> torch.Tensor:
    base = torch.load(path, map_location="cpu", mmap=True)
    if slice_info is None:
        return base
    else:
        return base[*slice_info]

def save_tensor(tensor: torch.Tensor, path: str | os.PathLike, slice_info: SliceInfo | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    if slice_info is None:
        torch.save(tensor, path)
        return
    
    src_tensor = get_file_tensor(path, slice_info)
    src_tensor[...] = tensor
    
    with open(path, "r+b") as f:
        f.flush()
        os.fsync(f.fileno())

@dataclass
class TensorPointer:
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
    
    def to_fs(self, path: str | os.PathLike, slice_info: SliceInfo | None = None) -> None:
        save_tensor(self.tensor, path, slice_info)
    
    def load(self, path: str | os.PathLike, slice_info: SliceInfo | None = None) -> None:
        src_tensor = get_file_tensor(path, slice_info)
        if src_tensor.shape != self.tensor.shape:
            raise Exception(f"Attempted to load a {src_tensor.shape} file tensor into a {self.tensor.shape} GPU tensor.")
        self.tensor[...] = src_tensor
    
    def __enter__(self) -> "TensorPointer":
        return self
    
    def __exit__(self, exc_t, exc_v, exc_tb) -> None:
        self.__parent._remove_proxy(self)
        self.__tensor = None
        self.__start = 0
        self.__length = 0

memory_manager_cache: "MemoryManager | bool" = True
max_mem_options: int | Iterable[int] = 3000
class MemoryManager:
    # Records must always be ordered by start index
    __records: list[TensorPointer]
    __max_mem: int
    __tensor: torch.Tensor | None
    
    def __init__(self, tensor: torch.Tensor, max_mem: int) -> None:
        self.__records = []
        self.__max_mem = max_mem
        self.__tensor = tensor
    
    @property
    def max_mem(self) -> int:
        return self.__max_mem
    
    @property
    def __present_tensor(self) -> torch.Tensor:
        if self.__tensor is None:
            raise Exception()
        
        return self.__tensor
    
    @staticmethod
    def get() -> "MemoryManager":
        global memory_manager_cache
        global max_mem_options
        if not isinstance(memory_manager_cache, bool): return memory_manager_cache
        if memory_manager_cache is False:
            raise Exception("Cannot create MemoryManager more than once")
        
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available")
        
        torch.set_default_device("cuda:0")
        total_memory = torch.cuda.get_device_properties(0).total_memory
        max_mem_allowed = total_memory // 8 // 2
        
        max_mem: int | None = None
        if isinstance(max_mem_options, int):
            if max_mem_options <= max_mem_allowed:
                max_mem = max_mem_options
        else:
            for option in sorted(max_mem_options, reverse=True):
                if option <= max_mem_allowed:
                    max_mem = option
                    break
        
        if max_mem is None:
            raise Exception(f"Not enough GPU memory.")
        
        tensor = torch.empty(max_mem, dtype=element_type)
        
        result = MemoryManager(tensor, max_mem)
        memory_manager_cache = result
        return result
    
    def empty(self, shape: tuple[int, ...]) -> TensorPointer:
        length = reduce(lambda a, b: a * b, shape, 1)
        rec_len = len(self.__records)
        
        for i in range(-1, rec_len):
            min_pos = self.__records[i].end if i >= 0 else 0
            max_pos = self.__records[i + 1].start if i + 1 < rec_len else self.__max_mem
            if length <= max_pos - min_pos:
                new_record = TensorPointer(min_pos, length, self, self.__present_tensor[min_pos : min_pos + length].view(*shape))
                self.__records.insert(i + 1, new_record)
                return new_record
        
        raise Exception(f"Not enough memory for new tensor of shape {shape}")
    
    def load(self, path: str | os.PathLike, slice_info: SliceInfo | None = None) -> TensorPointer:
        src_tensor = get_file_tensor(path, slice_info)
        result = self.empty(src_tensor.shape)
        result.tensor[...] = src_tensor
        return result
    
    def _remove_proxy(self, proxy: TensorPointer) -> None:
        for i in range(len(self.__records)):
            if id(i) == id(proxy):
                del self.__records[i]
                return
    
    def __enter__(self) -> "MemoryManager":
        return self
    
    def __exit__(self, exc_t, exc_v, exc_tb) -> None:
        global memory_manager_cache
        memory_manager_cache = False
        
        del self.__tensor
        self.__tensor = None
        torch.cuda.empty_cache()

__all__ = ["MemoryManager", "save_tensor"]

