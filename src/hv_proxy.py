from pathlib import Path
from collections.abc import Iterable, Sequence

import torch

from device import default_device
from fs_organization import FsOrganizer
from mutag import IdsToLabelsMapping
from constants import D, m

def f1(id: int, ids_to_labels: IdsToLabelsMapping, root: FsOrganizer) -> "HVProxy":
	label = ids_to_labels.label_of(id)
	path = root.hv_encoding_of(id)
	return HVProxy(id, label, path)

# Represents a proxy to a HV that is already defined in a file
class HVProxy:
	__id: int
	__label: int
	__path: Path
	
	def __init__(self, id: int, label: int, path: Path):
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
	
	def get_hv_on(self, out: torch.Tensor) -> torch.Tensor:
		tmp = torch.load(self.__path, map_location="cpu")
		out[...] = tmp
		return out

# Requires defined mapping from ids to labels
def iter_from_fs(root: FsOrganizer, ids: Iterable[int]) -> Iterable[HVProxy]:
	ids_to_labels = IdsToLabelsMapping.from_fs(root.ids_to_labels)
	return ( f1(id, ids_to_labels, root) for id in ids )

def iter_to_batch(proxies: Sequence[HVProxy, ...]) -> torch.Tensor:
	length = len(proxies)
	result = torch.zeros(length, D, m, m, dtype=torch.complex64, device=default_device)
	
	for i, proxy in enumerate(proxies):
		proxy.get_hv_on(result[i])
	
	return result
