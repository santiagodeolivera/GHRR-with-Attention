from pathlib import Path
import json
from typing import TypeVar
from pathlib import Path
import csv
from abc import ABC
import itertools

from operation_manager import OperationManagerRecord
from constants import DataType, D, m

# TODO: make sure tensor names are unique

T = TypeVar("T")

type Seq[T] = tuple[T, ...]

class OperationTemplate(ABC):
    def to_rows(self) -> Iterable[Seq[str]]: ...

class LeafTemplate(OperationTemplate):
    __row: Seq[str]
    
    def __init__(self, *values: str):
        self.__row = values
    
    def to_rows(self) -> Iterable[Seq[str]]:
        return (self.__row,)

class BranchTemplate(OperationTemplate):
    __children: Seq[OperationTemplate]
    
    def __init__(self, *children: OperationTemplate):
        self.__children = children
    
    def to_rows(self) -> Iterable[Seq[str]]:
        return itertools.chain.from_iterable(c.to_rows() for c in self.__children)

class NameSet:
    __names: set[str]
    
    def __init__(self):
        self.__names = set()
    
    def register(self, name: str) -> None:
        if name in self.__names:
            raise Exception(f"Duplicate tensor name: {repr(name)}")
        
        self.__names.add(name)

names = NameSet()

class TmpGenerator:
    __n: int
    
    def __init__(self):
        self.__n = 0
    
    def new(self, n: int) -> Seq[str]:
        start = self.__n
        stop = self.__n + n
        
        result = tuple(f"tmp/{x}" for x in range(start, stop))
        
        self.__n = stop
        return result

tmp_gen = TmpGenerator()

def randn(out: str, shape: tuple[int, ...], dtype: DataType) -> OperationTemplate:
    names.register(out)
    return LeafTemplate(out, "randn", json.dumps(shape), dtype.name)

def add(out: str, id1: str, id2: str) -> OperationTemplate:
    names.register(out)
    return LeafTemplate(out, "add", id1, id2)

def sum_(out: str, id1: str, unit_dims: int) -> OperationTemplate:
    names.register(out)
    return LeafTemplate(out, "sum", id1, str(unit_dims))

def matmul(out: str, id1: str, id2: str) -> OperationTemplate:
    names.register(out)
    return LeafTemplate(out, "matmul", id1, id2)

def normalize(out: str, id1: str) -> OperationTemplate:
    names.register(out)
    return LeafTemplate(out, "normalize", id1)

def adjoint(out: str, id1: str) -> OperationTemplate:
    names.register(out)
    return LeafTemplate(out, "adjoint", id1)

def real(out: str, id1: str) -> OperationTemplate:
    names.register(out)
    return LeafTemplate(out, "real", id1)

def softmax(out: str, id1: str) -> OperationTemplate:
    names.register(out)
    return LeafTemplate(out, "softmax", id1)

def swapdims(out: str, id1: str, dim1: int, dim2: int) -> OperationTemplate:
    names.register(out)
    return LeafTemplate(out, "swapdims", id1, str(dim1), str(dim2))

def slicerange(out: str, id1: str, dim: int, min_v: int, , max_v: int) -> OperationTemplate:
    names.register(out)
    return LeafTemplate(out, "slicerange", id1, str(dim), str(min_v), str(max_v))

def comment(s: str) -> OperationTemplate:
    return ("#", s)

def random_hvs(out: str, length: int) -> OperationTemplate:
    dtype = DataType.complex64
    
    return BranchTemplate(
        comment(f"#{out} = random_hvs({length}, D = {D}, m = {m}, dtype = {dtype.name})"),
        randn(out, (length, D, m, m), dtype)
    )

def sum_hvs(out: str, id1: str) -> OperationTemplate:
    tmp = tmp_gen.new(2)
    return BranchTemplate(
        comment(f"#{out} = sum_hvs(#{id1})"),
        swapdims(tmp[0], id1, -4, -1),
        sum_(tmp[1], tmp[0], 1),
        swapdims(out, tmp[1], -4, -1)
    )

def fn1(out: str, pos_enc: str, enc: str, fn_name: str) -> OperationTemplate:
    tmp = tmp_gen.new(2)
    return BranchTemplate(
        comment(f"#{out} = {fn_name}(#{pos_enc}, #{enc})"),
        matmul(tmp[0], pos_enc, enc),
        sum_hvs(tmp[1], tmp[0]),
        normalize(out, tmp[1])
    )

def query_from_encoded(out: str, pos_enc: str, enc: str) -> OperationTemplate:
    return fn1(out, pos_enc, enc, "query_from_encoded")

def key_from_encoded(out: str, enc1: str, enc2: str, pos_enc2: str) -> OperationTemplate:
    tmp = tmp_gen.new(4)
    return BranchTemplate(
        comment(f"#{out} = key_from_encoded(#{enc1}, #{enc2}, #{pos_enc2})"),
        matmul(tmp[0], enc2, pos_enc2),
        adjoint(tmp[1], tmp[0]),
        matmul(tmp[2], tmp[1], enc1),
        sum_hvs(tmp[3], tmp[2]),
        normalize(out, tmp[3])
    )

def value_from_encoded(out: str, pos_enc: str, enc: str) -> OperationTemplate:
    return fn1(out, pos_enc, enc, "value_from_encoded")

def templates_to_file(path: Path, templates: Seq[OperationTemplate]) -> None:
    with open(path, "w") as file:
        writer = csv.writer(file, delimiter=";", quoting=csv.QUOTE_NONE)
        writer.write_rows(templates)

