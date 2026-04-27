import json
from random import shuffle as random_shuffle
from typing import TypeVar, Callable, Any, Sequence, Iterable
from pathlib import Path
from math import sqrt
from functools import reduce
from get_args import get_arg
from tudataset import get_dataset_info

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
    num_graphs = get_dataset_info().num_graphs
    ids: list[int] = list(range(num_graphs))
    proportion = get_train_and_test_sets_proportion()
    
    train_ids, test_ids = proportional_split(ids, proportion)
    
    with open(path, "w") as file:
        json.dump({"train": train_ids, "test": test_ids}, file)

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
    if n <= 0:
        return "<empty list>"
    
    avg = sum(numbers) / n
    std = sqrt(sum((x - avg)**2 for x in numbers) / n)
    
    v1 = "element" if n == 1 else "elements"
    return f"{avg*100:.02f}% \u00B1 {std*100:.02f}% from {n} {v1}"

def get_size(shape: tuple[int, ...]) -> int:
    return reduce(lambda a, b: a * b, shape, 1)

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

def clamp(v: float, min: float | None, max: float | None) -> float:
    if min is not None and v < min:
        return min
    elif max is not None and v > max:
        return max
    else:
        return v

# TODO: Use this function wherever possible
def split_interval(start: int, end: int, size: int) -> Iterable[tuple[int, int]]:
    while start < end:
        mid_end = min(end, start + size)
        yield (start, mid_end)
        start = mid_end

class AccAvg:
    __sum: float
    __num: int
    __default: float
    
    def __init__(self, *, default: float) -> None:
        self.__sum = 0
        self.__num = 0
        self.__default = default
    
    def add(self, el: float) -> None:
        self.__sum += el
        self.__num += 1
    
    def add_all(self, it: Iterable[float]) -> None:
        elements = tuple(it)
        self.__sum += sum(elements)
        self.__num += len(elements)
    
    def get(self) -> float:
        if self.__num <= 0:
            return self.__default
        
        return self.__sum / self.__num

