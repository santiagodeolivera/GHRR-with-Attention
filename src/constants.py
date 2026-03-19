from builtins import slice
from typing import Any

import torch

class DataType:
    name: str
    value: Any
    size: int
    is_complex: bool
    
    complex64: "DataType"
    float32: "DataType"
    
    def __init__(self, name: str, value: Any, size: int, is_complex: bool):
        self.name = name
        self.value = value
        self.size = size
        self.is_complex = is_complex
    
    @staticmethod
    def get_by_name(name: str) -> "DataType":
        try:
            return next(v for v in _data_types if v.name == name)
        except StopIteration:
            raise ValueError(f"Invalid DataType name: {repr(name)}")

DataType.complex64 = DataType("complex64", torch.complex64, 8, True)
DataType.float32 = DataType("float32", torch.float32, 4, False)

_data_types: set[DataType] = { DataType.complex64, DataType.float32 }

type SliceInfo = tuple[slice[int, int, int], ...]

