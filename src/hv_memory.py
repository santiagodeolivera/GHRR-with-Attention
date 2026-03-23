import torch
from pathlib import Path

from utils import CheckpointContext
from hv_functions import normalize
from gpu_management import DataType, TensorProxy, TensorFunctionsManager
from constants import D, m

def get_random_hvs(manager: TensorFunctionsManager, file_path: Path, length: int) -> TensorProxy:
    mid_result = TensorProxy.get_if_exists(file_path)
    if mid_result is not None:
        return mid_result
    
    return manager.randn((length, D, m, m), file_path, DataType.complex64)

