from pathlib import Path
from collections.abc import Iterable, Sequence

import torch

from fs_organization import FsOrganizer
from tudataset import get_ids_to_labels_mapping
from constants import D, m
from gpu_management.data_type import DataType
from mmap_tensors import MmapTensors

def f1(id: int, ids_to_labels: tuple[int, ...], root: FsOrganizer) -> "HVProxy":
    label = ids_to_labels[id]
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
    
    def get_hv(self, *, out: torch.Tensor | None = None) -> torch.Tensor:
        result = MmapTensors.read_if_exists(self.__path)
        if result is None:
            raise Exception(f"No tensor in {self.__path}")
        
        if out is not None:
            out[...] = result
            return out
        
        return result

# Requires defined mapping from ids to labels
def iter_from_fs(root: FsOrganizer, ids: Iterable[int]) -> Iterable[HVProxy]:
    ids_to_labels = get_ids_to_labels_mapping(root.ids_to_labels)
    return ( f1(id, ids_to_labels, root) for id in ids )

def iter_to_batch(proxies: Sequence[HVProxy], out: torch.Tensor | None = None) -> torch.Tensor:
    length = len(proxies)
    
    result = out if out is not None else torch.empty(length, D, m, m, dtype=torch.complex64)
    
    for i, proxy in enumerate(proxies):
        proxy.get_hv(out=result[i])
    
    return result

