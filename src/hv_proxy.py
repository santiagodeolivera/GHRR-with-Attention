from pathlib import Path

import torch

from device import default_device

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
