import torch
from pathlib import Path

from gpu_management.data_type import DataType
from gpu_management.tensor_functions import TensorFunctionsManager
from constants import D, m
from time_ import Timer
from mmap_tensors import MmapTensors

def get_random_hvs(manager: TensorFunctionsManager, path: Path, length: int) -> torch.Tensor:
    if MmapTensors.exists(path):
        return MmapTensors.read_unsafe(path)
    else:
        result = MmapTensors.new_unsafe(path, (length, D, m, m), DataType.complex64)
        
        timer = Timer(f"Create random HV tensor in {path}")
        manager.randn((length, D, m, m), DataType.complex64, out=result)
        timer.end()
        
        return result
