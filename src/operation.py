from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Sequence
from functools import cached_property
import json
import csv

from tensor_functions import TensorFunctionsManager
from tensor_proxy import TensorProxy
from constants import DataType
from utils import block

@dataclass
class Operation:
    output_name: str
    input_names: tuple[str, ...]
    fn: Callable[..., TensorProxy]

tutorial_text_lines: list[str] = []

@block
def operation_builders() -> dict[str, Callable[[str, Sequence[str]], Operation]]:
    global tutorial_text_lines
    
    def str_to_int_tuple(v: str) -> tuple[int, ...]:
        v2 = json.loads(v)
        
        if not isinstance(v2, list):
            raise Exception()
        
        if not all(isinstance(n, int) for n in v2):
            raise Exception
        
        return tuple(v2)
    
    def str_to_data_type(v: str) -> DataType:
        return DataType.get_by_name(v)
    
    def str_to_int(v: str) -> int:
        return int(v)
    
    def randn(out: str, inputs: Sequence[str]) -> Operation:
        shape = str_to_int_tuple(inputs[0])
        data_type = str_to_data_type(inputs[1])
        return Operation(out, (), lambda out, fns: fns.randn(shape, out, data_type))
    tutorial_text_lines.append("<id_result> ; randn ; <json_shape> ; <dtype_name>")

    def addition(out: str, inputs: Sequence[str]) -> Operation:
        n1, n2 = inputs[0:2]
        return Operation(out, (n1, n2), lambda v1, v2, out, fns: fns.addition(v1, v2, out))
    tutorial_text_lines.append("<id_result> ; add ; <id_input_1> ; <id_input_1>")

    def summation(out: str, inputs: Sequence[str]) -> Operation:
        n1 = inputs[0]
        unit_dims = str_to_int(inputs[1])
        return Operation(out, (n1,), lambda v1, out, fns: fns.summation(v1, unit_dims, out))
    tutorial_text_lines.append("<id_result> ; sum ; <id_input> ; <unit_dims>")

    def matrix_mult(out: str, inputs: Sequence[str]) -> Operation:
        n1, n2 = inputs[0:2]
        return Operation(out, (n1, n2), lambda v1, v2, out, fns: fns.matrix_mult(v1, v2, out))
    tutorial_text_lines.append("<id_result> ; matmul ; <id_input_1> ; <id_input_2>")

    def normalize(out: str, inputs: Sequence[str]) -> Operation:
        n1 = inputs[0]
        return Operation(out, (n1,), lambda v1, out, fns: fns.normalize(v1, out))
    tutorial_text_lines.append("<id_result> ; normalize ; <id_input>")

    def adjoint(out: str, inputs: Sequence[str]) -> Operation:
        n1 = inputs[0]
        return Operation(out, (n1,), lambda v1, out, fns: fns.adjoint(v1, out))
    tutorial_text_lines.append("<id_result> ; adjoint ; <id_input>")

    def real(out: str, inputs: Sequence[str]) -> Operation:
        n1 = inputs[0]
        return Operation(out, (n1,), lambda v1, out, fns: fns.real(v1, out))
    tutorial_text_lines.append("<id_result> ; real ; <id_input>")

    def softmax(out: str, inputs: Sequence[str]) -> Operation:
        n1 = inputs[0]
        return Operation(out, (n1,), lambda v1, out, fns: fns.softmax(v1, out))
    tutorial_text_lines.append("<id_result> ; softmax ; <id_input>")

    def swap_dims(out: str, inputs: Sequence[str]) -> Operation:
        n1 = inputs[0]
        i1, i2 = tuple(str_to_int(v) for v in inputs[1:3])
        return Operation(out, (n1,), lambda v1, out, fns: fns.swap_dims(v1, i1, i2, out))
    tutorial_text_lines.append("<id_result> ; swapdims ; <id_input> ; <dim1> ; <dim2>")
    
    def slice_range(out: str, inputs: Sequence[str]) -> Operation:
        n1 = inputs[0]
        dim, min_v, max_v = tuple(str_to_int(v) for v in inputs[1:4])
        return Operation(out, (n1,), lambda v1, out, fns: fns.slice_range(v1, dim, min_v, max_v, out))
    tutorial_text_lines.append("<id_result> ; slicerange ; <id_input> ; <dim> ; <min> ; <max>")
    
    return {
        "randn": randn,
        "add": addition,
        "sum": summation,
        "matmul": matrix_mult,
        "normalize": normalize,
        "adjoint": adjoint,
        "real": real,
        "softmax": softmax,
        "swapdims": swap_dims,
        "slicerange": slice_range
    }

tutorial_text = "".join(f"# ; {s}\n" for s in tutorial_text_lines)

def operation_from_csv_row(data: Sequence[str]) -> Operation | None:
    try:
        trimmed_data = tuple(s.strip() for s in data)
        
        if trimmed_data[0] == "#":
            return None
        
        output_name = trimmed_data[0]
        
        function_type = trimmed_data[1]
        builder = operation_builders.get(function_type, None)
        if builder is None:
            raise Exception(f"Unknown operation builder type: {repr(function_type)}")
        
        return builder(output_name, trimmed_data[2:])
    except Exception as e:
        raise Exception(f"Failed to parse CSV row into operation: {repr(tuple(trimmed_data))}") from e

def operations_from_csv_file(path: Path) -> tuple[Operation, ...]:
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter=";", quoting=csv.QUOTE_NONE)
        mid_result = (operation_from_csv_row(row) for row in reader)
        return tuple(op for op in mid_result if op is not None)

__all__ = ["Operation", "operations_from_csv_file", "tutorial_text"]

