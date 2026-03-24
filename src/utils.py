import time
import os
import json
from random import shuffle as random_shuffle
from typing import TypeVar, TypeGuard, Callable, Any, Protocol, Sequence, Type
from pathlib import Path
from math import sqrt
from functools import reduce
from get_args import get_arg
from time_start import time_start
from datetime import datetime
from gpu_management import DataType
import shutil

import torch

# TODO: Find out where these functions should be located

T = TypeVar('T')

def check_int(v: Any) -> int:
    if not isinstance(v, int):
        raise ValueError(f"Expected an int")
    
    return v

def value_or_else(v: T | None, default_fn: Callable[[], T]) -> T:
    if v is not None:
        return v
    
    return default_fn()

def calc_time_difference(before: int, after: int):
    time_difference = (after - before) // 10_000_000

    if time_difference < 0:
        time_difference *= -1
    
    time_difference_int = time_difference // 100
    time_difference_dec = time_difference  % 100

    return f"{time_difference_int}.{time_difference_dec:02} s"

class Timer:
    start: int
    name: str
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.start = time.time_ns()
        
        self.__print(self.start, f"{self.name} -> START")
    
    def __print(self, time: int, msg: str) -> None:
        seconds = time / 1_000_000_000
        date = datetime.fromtimestamp(seconds)
        print(f"({date.hour:02}:{date.minute:02}, {calc_time_difference(time_start, time)} since program start) {msg}")
    
    def end(self) -> None:
        end = time.time_ns()
        self.__print(end, f"{self.name} -> END (took {calc_time_difference(self.start, end)})")
    
    def error(self) -> None:
        end = time.time_ns()
        self.__print(end, f"{self.name} -> ERROR (took {calc_time_difference(self.start, end)})")

def torch_cantor_pairing(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.where(a >= 0, a * 2, a * (-2) - 1)
    b = torch.where(b >= 0, b * 2, b * (-2) - 1)
    v1 = (a + b) * (a + b + 1)
    return torch.div(v1, 2, rounding_mode="trunc") + b

def cantor_pairing(a: int, b: int) -> int:
    a = a * 2 if a >= 0 else a * -2 - 1
    b = b * 2 if b >= 0 else b * -2 - 1
    return (a + b) * (a + b + 1) // 2 + b

def commutative_cantor_pairing(a: int, b: int) -> int:
    if a > b: (a, b) = (b, a)
    return cantor_pairing(a, b)

def log(value: T, msg: str | None = None, show: bool | Callable[[T], Any] = False) -> T:
    if isinstance(show, bool):
        if show:
            print(f"{msg}: {value}" if msg is not None else value)
        else:
            if msg is not None:
                print(msg)
    else:
        v1 = show(value)
        print(f"{msg}: {v1}" if msg is not None else v1)
    
    return value

def print_tensor_struct(t: torch.Tensor) -> str:
    v1 = t.dtype
    v2 = ", ".join(str(x) for x in t.shape)
    return f"{v1}[{v2}]"

def find_unique_path(path_input: str | Path) -> Path:
    path: Path
    if isinstance(path_input, str):
        path = Path(path_input)
    else:
        path = path_input
    
    original_stem = path.stem
    suffix = path.suffix
    
    res = path
    i = 1
    while res.exists():
        new_name = f"{original_stem} ({i}){suffix}"
        res = path.with_name(new_name)
        i += 1
    
    return res

def get_range_tensor(upper_limit: int, *, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(tuple(range(upper_limit)), dtype=torch.int8, device=device)

def get_single_tensor(n: float, *, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(n, dtype=torch.float32, device=device)

# Changes the input parameter
def proportional_split(input: list[T], proportion: float) -> tuple[tuple[T, ...], tuple[T, ...]]:
    random_shuffle(input)
    pivot = round(proportion * len(input))
    
    left = tuple(input[:pivot])
    right = tuple(input[pivot:])
    return (left, right)

def get_train_and_test_sets_proportion() -> float:
    return get_arg("PROPORTION", "float")

def define_train_and_test_datasets(path: Path) -> None:
    ids: list[int] = list(range(188))
    proportion = get_train_and_test_sets_proportion()
    
    train_ids, test_ids = proportional_split(ids, proportion)
    json_data = json.dumps({"train": train_ids, "test": test_ids})
    
    path.write_text(json_data)

def get_train_and_test_datasets(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    json_data = path.read_text()
    
    raw_data = json.loads(json_data)
    
    if type(raw_data) != dict:
        raise Exception()
    if "train" not in raw_data:
        raise Exception()
    if "test" not in raw_data:
        raise Exception()
    
    train_ids = raw_data["train"]
    
    if type(train_ids) != list:
        raise Exception()
    if not all(isinstance(x, int) for x in train_ids):
        raise Exception()

    test_ids = raw_data["test"]
    
    if type(test_ids) != list:
        raise Exception()
    if not all(isinstance(x, int) for x in test_ids):
        raise Exception()
    
    return (tuple(train_ids), tuple(test_ids))

# Changes the input parameter
def take_random_from_list(input: list[T], number: int) -> tuple[T, ...]:
    random_shuffle(input)
    return tuple(input[:number])

def approximation(numbers: Sequence[float]) -> str:
    n = len(numbers)
    avg = sum(numbers) / n
    std = sqrt(sum((x - avg)**2 for x in numbers) / n)
    return f"{avg*100:.02f}% \u00B1 {std*100:.02f}%"

def get_size(shape: tuple[int, ...]) -> int:
    return reduce(lambda a, b: a * b, shape, 1)

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
        
        with open(json_path, "w") as file:
            metadata = json.load(file)
        shape = tuple(check_int(n) for n in metadata["shape"])
        size = get_size(shape)
        data_type = DataType.get_by_name(metadata["type"]).value
        
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

class ContiguousTensor:
    __original: torch.Tensor
    __contiguous: torch.Tensor
    __is_copy: bool
    
    def __init__(self, tensor: torch.Tensor) -> None:
        is_copy = tensor.is_contiguous()
        
        self.__tensor = tensor
        self.__contiguous = tensor if is_copy else tensor.contiguous()
        self.__is_copy = is_copy
    
    def __enter__(self) -> torch.Tensor:
        return self.__contiguous
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.__is_copy:
            self.__original[...] = self.__contiguous

