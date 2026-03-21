from pathlib import Path
import json
from typing import Any
import shutil

import torch

from memory import MemoryManager
from constants import SliceInfo, DataType
from utils import get_size

types_dict: set[tuple[str, Any]] = {
    ("complex64", torch.complex64),
    ("float32", torch.float32)
}

def type_by_name(name: str) -> Any:
    return next(v for k, v in types_dict if k == name)

def name_of_type(value: Any) -> Any:
    return next(k for k, v in types_dict if v == value)

def path_to_data(path: Path) -> Path:
    return path.with_suffix(".bin")

def path_to_json(path: Path) -> Path:
    return path.with_suffix(".json")

class TensorProxy:
    __path: Path
    
    def __init__(self, path: Path):
        self.__path = path.with_suffix("")
    
    @staticmethod
    def empty_override(shape: tuple[int, ...], path: Path, data_type: DataType) -> "TensorProxy":
        TensorProxy.remove(path)
        return TensorProxy.__empty(shape, path, data_type)
    
    @staticmethod
    def empty(shape: tuple[int, ...], path: Path, data_type: DataType) -> "TensorProxy":
        if TensorProxy.exists(path):
            raise Exception(f"TensorProxy in {path} already exists")
        
        return TensorProxy.__empty(shape, path, data_type)
        
    @staticmethod
    def __empty(shape: tuple[int, ...], path: Path, data_type: DataType) -> "TensorProxy":
        tensor_path = path_to_data(path)
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        
        json_path = path_to_json(path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        length = get_size(shape)
        with open(tensor_path, "wb") as f:
            for _ in range(length * data_type.size):
                f.write(b'\x00')
        
        json_data = {
            "shape": shape,
            "type": data_type.name
        }
        
        with open(json_path, "w") as f:
            json.dump(json_data, f)
        
        return TensorProxy(path)
    
    @property
    def __data_path(self) -> Path:
        return path_to_data(self.__path)
    
    @property
    def __json_path(self) -> Path:
        return path_to_json(self.__path)
    
    @property
    def __json(self) -> Any:
        src = self.__json_path
        
        with open(src, "r") as f:
            return json.load(f)
    
    @property
    def shape(self) -> tuple[int, ...]:
        data = self.__json["shape"]
        
        if type(data) == list and all(type(n) == int for n in data):
            return tuple(data)
        
        raise Exception(f"Json has invalid shape information")
    
    @property
    def data_type(self) -> Any:
        data = self.__json["type"]
        
        if not isinstance(data, str):
           raise Exception(f"Json has invalid data type information")
        
        return DataType.get_by_name(data)

    @property
    def size(self) -> int:
        return get_size(self.shape)
    
    def tensor(self) -> torch.Tensor:
        return torch.from_file(str(self.__data_path), size=self.size, shared=True, \
            dtype=self.data_type.value).view(self.shape)
    
    @staticmethod
    def exists(path: Path) -> bool:
        data_path = path_to_data(path)
        json_path = path_to_json(path)
        
        return data_path.exists() and data_path.is_file() and json_path.exists() and json_path.is_file()
    
    @staticmethod
    def remove(path: Path, *, ignore_errors: bool = True) -> None:
        shutil.rmtree(path_to_data(path), ignore_errors=ignore_errors)
        shutil.rmtree(path_to_json(path), ignore_errors=ignore_errors)
    
    @staticmethod
    def get_if_exists(path: Path) -> "TensorProxy | None":
        if not TensorProxy.exists(path):
            return None
        
        return TensorProxy(path)

__all__ = ["TensorProxy"]

