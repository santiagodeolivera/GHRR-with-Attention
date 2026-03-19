from pathlib import Path
from dataclasses import dataclass
from typing import Callable

from tensor_functions import TensorFunctionsManager
from tensor_proxy import TensorProxy
from constants import DataType

@dataclass
class Operation:
    output_name: str
    input_names: tuple[str, ...]
    fn: Callable[..., TensorProxy]

class OperationManager:
    __root: Path
    __fns: TensorFunctionsManager
    __tensors: dict[str, TensorProxy]
    __operations: list[Operation]
    __tmp_counter: int
    
    def __init__(self, root: Path, fns: TensorFunctionsManager) -> None:
        self.__root = root
        self.__fns = fns
        self.__tensors = dict()
        self.__operations = list()
        self.__tmp_counter = 0
    
    def __execute(self, op: Operation, override: bool) -> None:
        output_path = self.__root / op.output_name
        result: TensorProxy
        
        if not override:
            v1 = TensorProxy.get_if_exists(output_path)
            if v1 is not None:
                result = v1
            else:
                inputs = tuple(self.__tensors[name] for name in op.input_names)
                result = (op.fn)(*inputs, output_path)
        else:
            TensorProxy.remove(output_path)
            inputs = tuple(self.__tensors[name] for name in op.input_names)
            result = (op.fn)(*inputs, output_path)
        
        self.__tensors[op.output_name] = result
    
    def execute_all(self, override: bool = False) -> None:
        for op in self.__operations:
            self.__execute(op, override)
    
    def get_tensor(self, name: str) -> TensorProxy:
        return self.__tensors[name]
    
    def set_tmp_names(self, n: int) -> tuple[str, ...]:
        start = self.__tmp_counter
        result = tuple(f"tmp/{x}" for x in range(start, start + n))
        self.__tmp_counter += n
        return result
    
    def add_randn(self, shape: tuple[int, ...], out: str, data_type: DataType):
        op = Operation(out, (), lambda out: self.__fns.randn(shape, out, data_type))
        self.__operations.append(op)
    
    def add_addition(self, n1: str, n2: str, out: str):
        op = Operation(out, (n1, n2), lambda v1, v2, out: self.__fns.addition(v1, v2, out))
        self.__operations.append(op)
    
    def add_summation(self, n1: str, unit_dims: int, out: str):
        op = Operation(out, (n1,), lambda v1, out: self.__fns.summation(v1, unit_dims, out))
        self.__operations.append(op)
    
    def add_matrix_mult(self, n1: str, n2: str, out: str):
        op = Operation(out, (n1, n2), lambda v1, v2, out: self.__fns.matrix_mult(v1, v2, out))
        self.__operations.append(op)
    
    def add_normalize(self, n1: str, out: str):
        op = Operation(out, (n1,), lambda v1, out: self.__fns.normalize(v1, out))
        self.__operations.append(op)
    
    def add_adjoint(self, n1: str, out: str):
        op = Operation(out, (n1,), lambda v1, out: self.__fns.adjoint(v1, out))
        self.__operations.append(op)

    def add_real(self, n1: str, out: str):
        op = Operation(out, (n1,), lambda v1, out: self.__fns.real(v1, out))
        self.__operations.append(op)

    def add_softmax(self, n1: str, out: str):
        op = Operation(out, (n1,), lambda v1, out: self.__fns.softmax(v1, out))
        self.__operations.append(op)
    
    def add_swap_dims(self, n1: str, out: str):
        op = Operation(out, (n1,), lambda v1, out: self.__fns.softmax(v1, out))
        self.__operations.append(op)

