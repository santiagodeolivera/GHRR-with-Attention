import json
from typing import Iterable, Callable, Sequence
from math import sqrt
from pathlib import Path

from fs_organization import FsOrganizer
from utils import TestResultData

def approximation(numbers: Sequence[float]) -> str:
	n = len(numbers)
	avg = sum(numbers) / n
	std = sqrt(sum((x - avg)**2 for x in numbers) / n)
	return f"{avg*100:.02f}% \u00B1 {std*100:.02f}%"

def f1(root: FsOrganizer, instance_dir: str) -> ConfusionMatrix:
	root.config.result_file = instance_dir
	result_file = root.test_results
	
	return TestResultData.from_fs(result_file).to_conf_matrix()
	
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