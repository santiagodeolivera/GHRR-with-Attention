from pathlib import Path
import json

import torch

from memory import TensorPointer, MemoryManager
from constants import element_type, SliceInfo, element_size
from utils import get_size

class TensorProxy:
    __path: Path
    
    def __init__(self, path: Path):
        self.__path = path.with_suffix("")
    
    @staticmethod
    def empty(shape: tuple[int, ...], path: Path) -> "TensorProxy":
        tensor_path = path.with_suffix(".bin")
        shape_path = path.with_suffix(".json")
        
        length = get_size(shape)
        with open(tensor_path, "wb") as f:
            for _ in range(length * element_size):
                f.write(b'\x00')
        
        with open(shape_path, "w") as f:
            json.dump(shape, f)
        
        return TensorProxy(path)
    
    @property
    def __data_path(self) -> Path:
        return self.__path.with_suffix(".bin")
    
    @property
    def __shape_path(self) -> Path:
        return self.__path.with_suffix(".json")
    
    @property
    def shape(self) -> tuple[int, ...]:
        src = self.__shape_path
        
        with open(src, "r") as f:
            data = json.load(f)
        
        if type(data) == list and all(type(n) == int for n in data):
            return tuple(data)
        
        raise Exception(f"File {src} has invalid shape information")
    
    @property
    def size(self) -> int:
        return get_size(self.shape)
    
    def tensor(self) -> torch.Tensor:
        return torch.from_file(str(self.__data_path), size=self.size, shared=True, dtype=element_type).view(self.shape)

__all__ = ["TensorProxy"]

