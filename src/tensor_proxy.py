from pathlib import Path
import json

import torch

from memory import MemoryManager
from constants import element_type, SliceInfo, element_size
from utils import get_size

def path_to_data(path: Path) -> Path:
    return path.with_suffix(".bin")

def path_to_shape(path: Path) -> Path:
    return path.with_suffix(".json")

class TensorProxy:
    __path: Path
    
    def __init__(self, path: Path):
        self.__path = path.with_suffix("")
    
    @staticmethod
    def empty(shape: tuple[int, ...], path: Path) -> "TensorProxy":
        if TensorProxy.exists(path):
            raise Exception(f"TensorProxy in {path} already exists")
        
        tensor_path = path_to_data(path)
        shape_path = path_to_shape(path)
        
        length = get_size(shape)
        with open(tensor_path, "wb") as f:
            for _ in range(length * element_size):
                f.write(b'\x00')
        
        with open(shape_path, "w") as f:
            json.dump(shape, f)
        
        return TensorProxy(path)
    
    @property
    def __data_path(self) -> Path:
        return path_to_data(self.__path)
    
    @property
    def __shape_path(self) -> Path:
        return path_to_shape(self.__path)
    
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
    
    @staticmethod
    def exists(path: Path) -> bool:
        data_path = path_to_data(path)
        shape_path = path_to_shape(path)
        
        return data_path.exists() and data_path.is_file() and shape_path.exists() and shape_path.is_file()
    
    @staticmethod
    def get_if_exists(path: Path) -> "TensorProxy | None":
        if TensorProxy.exists(path):
            return None
        
        return TensorProxy(path)

__all__ = ["TensorProxy"]

