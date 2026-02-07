import time
import os
import json
from random import shuffle as random_shuffle
from typing import TypeVar, TypeGuard, Callable, Any, Protocol
from pathlib import Path

import torch

from device import default_device

# TODO: Find out where these functions should be located

T = TypeVar('T')

def not_none(v: T | None) -> TypeGuard[T]:
	return v is not None

def is_bool(v: T | bool) -> TypeGuard[bool]:
	return type(v) == bool

def is_str(v: T | str) -> TypeGuard[str]:
	return type(v) == str

def value_or(v: T | None, default: T) -> T:
	return v if not_none(v) else default

def check_int(v: Any) -> int:
	t = type(v)
	if t != int:
		raise ValueError(f"Expected an int, got {t}")
	
	return v

def value_or_else(v: T | None, default_fn: Callable[[], T]) -> T:
	if not_none(v):
		return v
	
	return default_fn()

def calc_time_difference(before: int, after: int):
	time_difference = (after - before) // 10_000_000

	if time_difference < 0:
		time_difference *= -1
	
	time_difference_int = time_difference // 100
	time_difference_dec = time_difference  % 100

	return f"{time_difference_int}.{time_difference_dec:02} s"

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
	if is_bool(show):
		if show:
			print(f"{msg}: {value}" if msg is not None else value)
		else:
			if msg is not None:
				print(msg)
	else:
		v1 = show(value)
		print(f"{msg}: {v1}" if msg is not None else v1)
	
	return value

class ICheckpointContext(Protocol):
	def print(self, msg: str) -> None: ...
	
	def log(self, msg: str, value: T) -> T:
		self.print(msg)
		return value

class CheckpointContext(ICheckpointContext):
	name: str
	start_time: int
	last_time: int
	
	def __init__(self, name: str, *, msg: str | None = None):
		self.start_time = time.time_ns()
		self.last_time = self.start_time
		
		self.name = name
		
		if msg is not None:
			self.print(msg)
		
	def print(self, msg: str) -> None:
		current_time = time.time_ns()
		diff_from_start = calc_time_difference(self.start_time, current_time)
		diff_from_last = calc_time_difference(self.last_time, current_time)
		self.last_time = current_time
		print()
		print(f"Checkpoint context: {self.name}")
		print(diff_from_last, "since last checkpoint")
		print(diff_from_start, "since checkpoint context definition")
		print(msg)

class VoidCheckpointContext(ICheckpointContext):
	def print(self, msg: str) -> None:
		pass

def print_tensor_struct(t: torch.Tensor) -> str:
	v1 = t.dtype
	v2 = ", ".join(str(x) for x in t.shape)
	return f"{v1}[{v2}]"

def find_unique_path(path_input: str | Path) -> Path:
	path: Path
	if is_str(path_input):
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

def get_range_tensor(upper_limit: int) -> torch.Tensor:
	return torch.tensor(tuple(range(upper_limit)), dtype=torch.int8, device=default_device)

def get_single_tensor(n: float) -> torch.Tensor:
	return torch.tensor(n, dtype=torch.float32, device=default_device)

# Changes the input parameter
def proportional_split(input: list[T], proportion: float) -> tuple[tuple[T, ...], tuple[T, ...]]:
	random_shuffle(input)
	pivot = round(proportion * len(input))
	
	left = tuple(input[:pivot])
	right = tuple(input[pivot:])
	return (left, right)

def get_train_and_test_sets_proportion() -> float | None:
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
	if any(type(x) != int for x in train_ids):
		raise Exception()

	test_ids = raw_data["test"]
	
	if type(test_ids) != list:
		raise Exception()
	if any(type(x) != int for x in test_ids):
		raise Exception()
	
	return (train_ids, test_ids)
