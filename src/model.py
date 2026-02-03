from pathlib import Path

import torch

from constants import D, m
from hv_proxy import HVProxy
from device import default_device
from hv_functions import normalize, normalized_similarity

def get_class_hv_from_file(path: Path) -> tuple[int, torch.Tensor]:
	id = int(path.stem)
	hv = torch.load(path, map_location=default_device)
	return (id, hv)

class Model:
	__classes: dict[int, torch.Tensor]
	
	def __init__(self, classes: dict[int, torch.Tensor]):
		self.__classes = classes
	
	def to_fs(self, root_dir: Path) -> None:
		for label, class_hv in self.__classes:
			torch.save(class_hv, root_dir / f"{label}.pt")
	
	@staticmethod
	def from_fs(self, root_dir: Path) -> "Model":
		class_hv_iter = ( get_class_hv_from_file(path) for path in root_dir.glob("*.pt") )
		classes: dict[int, torch.Tensor] = dict(class_hv_iter)
		return Model(classes)
	
	@staticmethod
	def train(training_set: Iterable[HVProxy]) -> "Model":
		unnormalized_class_hvs: dict[int, torch.Tensor] = dict()
		
		for proxy in training_set:
			label = proxy.label
			if label not in unnormalized_class_hvs:
				unnormalized_class_hvs[label] = torch.zeros(D, m, m)
			
			new_hv: torch.Tensor = proxy.get_hv()
			torch.add(unnormalized_class_hvs[label], new_hv, out=unnormalized_class_hvs[label])
			
			del new_hv
		
		classes = dict((k, normalize(v)) for k, v in unnormalized_class_hvs.items())
		
		return Model(classes)

	# test_batch: (x)D batch of HVs
	# returns: (x)D batch of integers
	def test(self, test_batch: torch.Tensor) -> torch.Tensor:
		res_shape = test_batch.shape[:-3]
		
		if not res_shape:
			min_distance: torch.Tensor | None = None
			closest_label: int = -1
			
			for label, class_hv in self.__classes:
				distance: torch.Tensor = normalized_similarity(test_batch, class_hv)
				
				if min_distance is None or distance.item() < min_distance.item():
					min_distance = distance
					closest_label = label
			
			if min_distance is None:
				raise Exception()
			
			return torch.tensor(closest_label)
		elif any(x == 0 for x in res_shape):
			return torch.zeros(res_shape)
		else:
			min_distances: torch.Tensor = torch.zeros(*res_shape)
			closest_labels: torch.Tensor = torch.zeros(*res_shape, dtype=np.int8)
			defined_vars: bool = False
			
			for label, class_hv in self.__classes:
				class_hv_2 = class_hv[*((None,) * len(res_shape)), ...].expand(*res_shape, -1, -1, -1)
				distances: torch.Tensor = normalized_similarity(test_batch, class_hv_2)
				
				if not defined_vars:
					min_distances = distances
					closest_labels = torch.full(res_shape, label, dtype=np.int8)
					defined_vars = True
					continue
				
				modify_result = distances < min_distances
				min_distances = torch.where(modify_result, distances, min_distances)
				closest_labels = torch.where(modify_result, label, closest_labels)
			
			if not defined_vars:
				raise Exception()
			
			return closest_labels
