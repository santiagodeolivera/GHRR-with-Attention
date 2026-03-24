import torch
from pathlib import Path

from gpu_management import DataType, TensorProxy, TensorFunctionsManager
from constants import D, m
from utils import Timer

def get_random_hvs(manager: TensorFunctionsManager, file_path: Path, length: int) -> TensorProxy:
    mid_result = TensorProxy.get_if_exists(file_path)
    if mid_result is not None:
        return mid_result
    
    timer = Timer()
    result = manager.randn((length, D, m, m), DataType.complex64, out=file_path)
    timer.msg(f"Random HV tensor in {file_path} created")
    return result

