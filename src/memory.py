from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch

from constants import SliceInfo, element_type, element_size
from utils import get_size

class MemoryManager:
    __max_mem: int
    __tensor: torch.Tensor | None
    
    def __init__(self, tensor: torch.Tensor, max_mem: int) -> None:
        self.__max_mem = max_mem
        self.__tensor = tensor
    
    @property
    def max_mem(self) -> int:
        return self.__max_mem
    
    @property
    def tensor(self) -> torch.Tensor:
        if self.__tensor is None:
            raise Exception()
        
        return self.__tensor
    
    @staticmethod
    def create(max_mem: int) -> "MemoryManager":
        global memory_manager_cache
        global max_mem_options
        
        total_memory = torch.cuda.get_device_properties(0).total_memory
        max_mem_allowed = total_memory // (element_size * 2)
        
        if max_mem > max_mem_allowed:
            raise Exception(f"Not enough GPU memory.")
        
        tensor = torch.empty(max_mem, dtype=element_type, device="cuda:0")
        
        result = MemoryManager(tensor, max_mem)
        return result
    
    def alloc_tensors(self, shapes0: Iterable[tuple[int, ...]]) -> tuple[torch.Tensor, ...]:
        shapes: Sequence[tuple[int, ...]] = tuple(shapes0)
        
        sizes = tuple(get_size(shape) for shape in shapes)
        required_size = sum(sizes)
        if required_size > self.max_mem:
            raise Exception("Not enough memory")
        
        result: list[torch.Tensor] = []
        start = 0
        for shape, size in zip(shapes, sizes):
            new_tensor = self.tensor[start : start + size].view(shape)
            result.append(new_tensor)
            start += size
        
        return tuple(result)
        
    
    def __enter__(self) -> "MemoryManager":
        return self
    
    def __exit__(self, exc_t, exc_v, exc_tb) -> None:
        del self.__tensor
        self.__tensor = None
        torch.cuda.empty_cache()

__all__ = ["MemoryManager"]

