from pathlib import Path

import torch

from constants import D, m
from hv_proxy import HVProxy
from device import default_device
from hv_functions import normalize

def get_class_hv_from_file(path: Path) -> tuple[int, torch.Tensor]:
	label = int(path.stem)
	hv = torch.load(path, map_location=default_device)
	return (label, hv)

class Model:
	__classes: dict[int, torch.Tensor]
	
	def __init__(self, classes: dict[int, torch.Tensor]):
		self.__classes = classes
	
	def to_fs(self, root_dir: Path):
		for label, class_hv in self.__classes:
			torch.save(class_hv, root_dir / f"{label}.pt")
	
	@staticmethod
	def from_fs(self, root_dir: Path) -> "Model":
		class_hv_iter = ( get_class_hv_from_file(path) for path in root_dir.glob("*.pt") )
		classes: dict[int, torch.Tensor] = dict(class_hv_iter)
		return Model(classes)
	
	@staticmethod
	def train(training_sets: dict[int, Iterable[HVProxy]]) -> "Model":
		classes: dict[int, torch.Tensor] = dict()
		
		for label, proxies in training_sets.items():
			sum_hv: torch.Tensor = torch.zeros(D, m, m)
			
			for proxy in proxies:
				new_hv: torch.Tensor = proxy.get_hv()
				torch.add(sum_hv, new_hv, out=sum_hv)
			
			class_hv: torch.Tensor = normalize(sum_hv)
			
			classes[label] = class_hv
		
		return Model(classes)
