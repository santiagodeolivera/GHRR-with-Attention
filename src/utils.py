import time
import os
import json
from random import shuffle as random_shuffle
from typing import TypeVar, TypeGuard, Callable, Any, Protocol, Sequence
from pathlib import Path
from math import sqrt
from functools import reduce

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
    
    def __init__(self) -> None:
        self.start = time.time_ns()
    
    def msg(self, s: str) -> None:
        end = time.time_ns()
        print(s, f"(took {calc_time_difference(self.start, end)})")

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
    proportion_str = os.environ.get("PROPORTION", None)
    if proportion_str is None:
        raise Exception("Proportion not defined")
    
    return float(proportion_str)

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

