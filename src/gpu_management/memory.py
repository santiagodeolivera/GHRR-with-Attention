from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Sequence, Any

import torch

from .data_type import DataType
from utils import get_size

class MemoryManager:
    __max_mem: int
    __tensor: torch.Tensor | None
    __data_type: DataType

    def __init__(self, tensor: torch.Tensor, max_mem: int, data_type: DataType) -> None:
        self.__max_mem = max_mem
        self.__tensor = tensor
        self.__data_type = data_type
    
    @property
    def max_mem(self) -> int:
        return self.__max_mem
    
    @property
    def data_type(self) -> DataType:
        return self.__data_type
    
    @property
    def tensor(self) -> torch.Tensor:
        if self.__tensor is None:
            raise Exception()
        
        return self.__tensor
    
    @staticmethod
    def create_two(max_mem: int) -> "tuple[MemoryManager, MemoryManager]":
        tensor1 = torch.empty(max_mem, dtype=torch.complex64, device="cuda:0")
        result1 = MemoryManager(tensor1, max_mem, DataType.complex64)
        
        tensor2 = torch.view_as_real(tensor1).view(-1)
        result2 = MemoryManager(tensor2, max_mem, DataType.float32)
        
        return (result1, result2)
    
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

