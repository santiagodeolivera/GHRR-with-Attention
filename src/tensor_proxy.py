from pathlib import Path
import json

import torch

from memory import TensorPointer, MemoryManager
from constants import element_type, SliceInfo

class TensorProxy:
    __path: Path
    
    def __init__(self, path: Path):
        self.__path = path.with_suffix("")
    
    @staticmethod
    def empty(shape: tuple[int, ...], path: Path) -> "TensorProxy":
        tensor_path = path.with_suffix(".pt")
        shape_path = path.with_suffix(".json")
        
        template = torch.tensor(0, dtype=element_type).broadcast_to(shape)
        torch.save(template, tensor_path)
        
        with open(shape_path, "w") as f:
            json.dump(shape, f)
        
        return TensorProxy(path)
    
    @property
    def __tensor_data(self) -> Path:
        return self.__path.with_suffix(".pt")
    
    @property
    def __shape_data(self) -> Path:
        return self.__path.with_suffix(".json")
    
    @property
    def shape(self) -> tuple[int, ...]:
        src = self.__shape_data
        
        with open(src, "r") as f:
            data = json.load(f)
        
        if type(data) == list and all(type(n) == int for n in data):
            return tuple(data)
        
        raise Exception(f"File {src} has invalid shape information")
    
    def load(self, slice_info: SliceInfo | None = None) -> TensorPointer:
        manager = MemoryManager.get()
        return manager.load(self.__path, slice_info)
    
    def save(self, tensor: TensorPointer, slice_info: SliceInfo | None = None) -> None:
        tensor.to_fs(self.__path, slice_info)

__all__ = ["TensorProxy"]

