import os
import json
from pathlib import Path
import shutil
from gpu_management.data_type import DataType
from utils import get_size, check_int

import torch

class MmapTensors:
    @staticmethod
    def get_paths(path: Path) -> tuple[Path, Path]:
        return path.with_suffix(".bin"), path.with_suffix(".json")
    
    @staticmethod
    def exists(path: Path) -> bool:
        data_path, json_path = MmapTensors.get_paths(path)
        return data_path.exists() and data_path.is_file() and json_path.exists() and json_path.is_file()
    
    @staticmethod
    def read_unsafe(path: Path) -> torch.Tensor:
        data_path, json_path = MmapTensors.get_paths(path)
        
        with open(json_path, "r") as file:
            metadata = json.load(file)
        shape = tuple(check_int(n) for n in metadata["shape"])
        size = get_size(shape)
        data_type = DataType.get_by_name(metadata["type"]).value
        
        print("DEBUG A:", data_path)
        return torch.from_file(str(data_path), size=size, shared=True, dtype=data_type).view(shape)
    
    @staticmethod
    def read_if_exists(path: Path) -> torch.Tensor | None:
        if not MmapTensors.exists(path):
            return None
        
        return MmapTensors.read_unsafe(path)
    
    @staticmethod
    def new_unsafe(path: Path, shape: tuple[int, ...], data_type: DataType) -> torch.Tensor:
        data_path, json_path = MmapTensors.get_paths(path)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        length = get_size(shape)
        data_path.touch(exist_ok=True)
        os.truncate(data_path, length * data_type.size)
        
        metadata = {
            "shape": shape,
            "type": data_type.name
        }
        
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        
        return MmapTensors.read_unsafe(path)
        
    @staticmethod
    def new_override(path: Path, shape: tuple[int, ...], data_type: DataType) -> torch.Tensor:
        data_path, json_path = MmapTensors.get_paths(path)
        shutil.rmtree(data_path, ignore_errors=True)
        shutil.rmtree(json_path, ignore_errors=True)
        
        return MmapTensors.new_unsafe(path, shape, data_type)
    
    @staticmethod
    def new_or_existing(path: Path, shape: tuple[int, ...], data_type: DataType) -> torch.Tensor:
        if MmapTensors.exists(path):
            result = MmapTensors.read_unsafe(path)
            dtype = data_type.value
            if result.dtype != dtype or result.shape != shape:
                raise ValueError(f"Expected a {shape}[{dtype}] tensor, not a {result.shape}[{result.dtype}] one")
            return result
        else:
            return MmapTensors.new_unsafe(path, shape, data_type)
