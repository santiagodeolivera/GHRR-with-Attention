from pathlib import Path

import torch

from device import default_device
from fs_organization import FsOrganizer
from mutag import get_ids_to_labels_mapping

def f1(id: int, ids_to_labels: tuple[int, ...], root: FsOrganizer) -> "HVProxy":
	label = ids_to_labels[id]
	path = root.hv_encoding_of(id)
	return HVProxy(id, label, path)

# Represents a proxy to a HV that is already defined
class HVProxy:
	__id: int
	__label: int
	__path: Path
	
	def __init__(id: int, label: int, path: Path):
		self.__id = id
		self.__label = label
		self.__path = path
	
	@property
	def id(self) -> int:
		return self.__id
	
	@property
	def label(self) -> int:
		return self.__label
	
	def get_hv(self) -> torch.Tensor:
		return torch.load(self.__path, map_location=default_device)
	
	# Requires defined mapping from ids to labels
	@staticmethod
	def iter_from_fs(root: FsOrganizer, ids: Iterable[int]) -> "Iterable[HVProxy]":
		ids_to_labels = get_ids_to_labels_mapping(root)
		return (f1(id, ids_to_labels, root) for id in ids)
