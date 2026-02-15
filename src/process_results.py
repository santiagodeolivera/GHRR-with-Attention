import json
from typing import Iterable, Callable, Sequence, Any
from math import sqrt
from pathlib import Path

from fs_organization import FsOrganizer

class ConfusionMatrix:
	__results: dict[tuple[int, int], int]
	__positive_label: int
	
	def __init__(self, results: dict[tuple[int, int], int], positive_label: int):
		self.__results = results
		self.__positive_label = positive_label
	
	@property
	def accuracy(self) -> float:
		return sum(v for (k, v) in self.__results.items() if k[0] == k[1]) \
			/ sum(self.__results.values())
	
	@property
	def precision(self) -> float:
		return sum(v for (k, v) in self.__results.items() if k[0] == k[1] and k[0] == self.__positive_label) \
			/ sum(v for (k, v) in self.__results.items() if k[1] == self.__positive_label)
	
	@property
	def recall(self) -> float:
		return sum(v for (k, v) in self.__results.items() if k[0] == k[1] and k[0] == self.__positive_label) \
			/ sum(v for (k, v) in self.__results.items() if k[0] == self.__positive_label)
	
	@property
	def f1(self) -> float:
		precision = self.precision
		recall = self.recall
		return 2 * precision * recall / (precision + recall)

def approximation(numbers: Sequence[float]) -> str:
	n = len(numbers)
	avg = sum(numbers) / n
	std = sqrt(sum((x - avg)**2 for x in numbers) / n)
	return f"{avg*100:.02f}% \u00B1 {std*100:.02f}%"

def check_result_data(v: Any) -> tuple[list[int], list[int], list[int]]:
	if type(v) != dict: raise ValueError()
	
	def f2(key: str) -> list[int]:
		try:
			value = v.get(key, None)
			if value is None: raise ValueError()
			if type(value) != list: raise ValueError()
			if any(type(x) != int for x in value): raise ValueError()
			return value
		except ValueError as e:
			v1 = json.dumps(key)
			raise Exception(f"Error while handling key {v1}") from e
	
	return tuple(f2(key) for key in ("ids", "expected", "result"))

def f1(root: FsOrganizer, instance_dir: str) -> ConfusionMatrix:
	root.config.result_file = instance_dir
	result_file = root.test_results
	json_data = result_file.read_text()
	raw_data = json.loads(json_data)
	_, expected_list, result_list = check_result_data(raw_data)
	
	conf_matrix_dict: dict[tuple[int, int], int] = dict()
	for i in range(len(expected_list)):
		expected = expected_list[i]
		result = result_list[i]
		key = (expected, result)
		
		if key not in conf_matrix_dict:
			conf_matrix_dict[key] = 0
		
		conf_matrix_dict[key] += 1
	
	return ConfusionMatrix(conf_matrix_dict, positive_label=1)
	
def process_results(instances: Iterable[str], out_file: str) -> Callable[[FsOrganizer], None]:
	def inner(root: FsOrganizer) -> None:
		conf_matrices = tuple(f1(root, instance_dir) for instance_dir in instances)
		
		metrics = ( \
			("accuracy",  lambda m: m.accuracy), \
			("precision", lambda m: m.precision), \
			("recall",    lambda m: m.recall), \
			("f1",        lambda m: m.f1) \
		)
		
		raw_result = dict()
		for name, fn in metrics:
			values = tuple(fn(m) for m in conf_matrices)
			approx_result = approximation(values)
			raw_result[name] = approx_result
		json_result = json.dumps(raw_result)
		(root.root / out_file).write_text(json_result)
	
	return inner

__all__ = ["process_results"]
