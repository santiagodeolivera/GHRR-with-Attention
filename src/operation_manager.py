from pathlib import Path
from dataclasses import dataclass
from typing import Callable
from functools import cached_property
import csv
import json

from tensor_functions import TensorFunctionsManager
from tensor_proxy import TensorProxy
from constants import DataType
from operation import Operation, operations_from_csv_file, tutorial_text

class OperationManagerRecord:
    __root: Path
    
    def __init__(self, root: Path) -> None:
        self.__root = root
    
    # Tensor storage
    @property
    def __tensors_path(self) -> Path:
        return self.__root / "tensors"
    
    # Operations storage
    @property
    def operations_path(self) -> Path:
        return self.__root / "operations.csv"
    
    # Progress storage
    @property
    def __progress_path(self) -> Path:
        return self.__root / "progress.json"
    
    def setup(self) -> None:
        self.__tensors_path.mkdir(parents=True, exist_ok=True)
        
        if not self.operations_path.exists():
            with open(self.operations_path, "w") as file:
                file.write(tutorial_text)
        
        if not self.__progress_path.exists():
            with open(self.__progress_path, "w") as file:
                json.dump(0, file)
    
    @property
    def progress(self) -> int:
        with open(self.__progress_path, "r") as file:
            res = json.load(file)
            if not isinstance(res, int):
                raise Exception()
            return res
    
    @progress.setter
    def progress(self, v: int) -> None:
        with open(self.__progress_path, "w") as file:
            return json.dump(v, file)
    
    @cached_property
    def operations(self) -> tuple[Operation, ...]:
        return operations_from_csv_file(self.operations_path)
    
    def get_tensor_path(self, key: str) -> Path:
        return self.__tensors_path / key
    
    def get_tensor(self, key: str) -> TensorProxy:
        proxy_path = self.get_tensor_path(key)
        result = TensorProxy.get_if_exists(proxy_path)
        if result is None:
            raise Exception(f"Invalid tensor proxy key: {repr(key)}")
        return result

class OperationManager:
    __fns: TensorFunctionsManager
    __record: OperationManagerRecord
    
    def __init__(self, root: Path, fns: TensorFunctionsManager) -> None:
        self.__fns = fns
        self.__record = OperationManagerRecord(root)
    
    def __execute(self, op: Operation) -> None:
        output_path = self.__record.get_tensor_path(op.output_name)
        result: TensorProxy
        
        TensorProxy.remove(output_path)
        inputs = tuple(self.__record.get_tensor(name) for name in op.input_names)
        result = (op.fn)(*inputs, output_path, self.__fns) # Automatically stores the tensor in the correct file
    
    def execute_all(self) -> None:
        operations = self.__record.operations
        
        progress = self.__record.progress
        while progress < len(operations):
            self.__execute(operations[progress])
            progress += 1
            self.__record.progress = progress
    
    def get_tensor(self, name: str) -> TensorProxy:
        return self.__record.get_tensor(name)

