from typing import Any

import torch

class DataType:
    name: str
    value: Any
    size: int
    is_complex: bool
    _to_real: "DataType | None"
    
    complex64: "DataType"
    float32: "DataType"
    
    def __init__(self, name: str, value: Any, size: int, is_complex: bool):
        self.name = name
        self.value = value
        self.size = size
        self.is_complex = is_complex
        self._to_real = None
    
    @staticmethod
    def get_by_name(name: str) -> "DataType":
        try:
            return next(v for v in _data_types if v.name == name)
        except StopIteration:
            raise ValueError(f"Invalid DataType name: {repr(name)}")
    
    @staticmethod
    def get_by_dtype(dtype: Any) -> "DataType":
        try:
            return next(v for v in _data_types if v.value == dtype)
        except StopIteration:
            raise ValueError(f"Invalid DataType value: {repr(dtype)}")
    
    def to_real(self) -> "DataType":
        if self._to_real is None:
            raise Exception()
        
        return self._to_real
    
    def is_real(self) -> bool:
        return self.to_real() is self

DataType.complex64 = DataType("complex64", torch.complex64, 8, True)
DataType.float32 = DataType("float32", torch.float32, 4, False)

DataType.complex64._to_real = DataType.float32
DataType.float32._to_real = DataType.float32

_data_types: set[DataType] = { DataType.complex64, DataType.float32 }

__all__ = ["DataType"]

