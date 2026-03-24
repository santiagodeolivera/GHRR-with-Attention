from pathlib import Path
from typing import Iterable

import torch

from constants import D, m
from hv_proxy import HVProxy
from hv_functions import UpperTensorFunctionsManager
from gpu_management import DataType
from utils import MmapTensors

def get_class_hv_from_file(path: Path) -> tuple[int, torch.Tensor]:
    id = int(path.stem)
    hv = MmapTensors.read_if_exists(path)
    if hv is None:
        raise Exception()
    return (id, hv)

def normalize_and_save(functions: UpperTensorFunctionsManager, hv: torch.Tensor, path: Path) -> torch.Tensor:
    result = MmapTensors.new_override(path, hv.shape, DataType.get_by_dtype(hv.dtype))
    functions.lower.normalize(hv, out=result)
    return result
    
class Model:
    __classes: dict[int, torch.Tensor]
    
    def __init__(self, classes: dict[int, torch.Tensor]):
        self.__classes = classes
    
    @staticmethod
    def from_fs(root_dir: Path) -> "Model":
        class_hv_iter = ( get_class_hv_from_file(path) for path in root_dir.glob("*.json") )
        classes: dict[int, torch.Tensor] = dict(class_hv_iter)
        return Model(classes)
    
    @staticmethod
    def train(functions: UpperTensorFunctionsManager, training_set: Iterable[HVProxy], out_dir: Path) -> "Model":
        unnormalized_class_hvs: dict[int, torch.Tensor] = dict()
        
        for proxy in training_set:
            label = proxy.label
            
            class_hv: torch.Tensor
            if label not in unnormalized_class_hvs:
                class_hv = torch.empty(D, m, m, dtype=torch.complex64)
                unnormalized_class_hvs[label] = class_hv
            else:
                class_hv = unnormalized_class_hvs[label]
            
            new_hv: torch.Tensor = proxy.get_hv()
            functions.lower.addition(class_hv, new_hv, out=class_hv)
            
            del new_hv
        
        classes: dict[int, torch.Tensor] = { k: normalize_and_save(functions, v, out_dir / str(k)) \
            for k, v in unnormalized_class_hvs.items() }
        
        return Model(classes)

    # test_batch: (x)D batch of HVs
    # returns: (x)D batch of integers
    def test(self, functions: UpperTensorFunctionsManager, test_batch: torch.Tensor) -> torch.Tensor:
        res_shape = test_batch.shape[:-3]
        
        if not res_shape:
            min_distance: torch.Tensor | None = None
            closest_label: int = -1
            
            for label, class_hv in self.__classes.items():
                distance: torch.Tensor = functions.normalized_similarity(test_batch, class_hv)
                
                if min_distance is None or distance.item() < min_distance.item():
                    min_distance = distance
                    closest_label = label
            
            if min_distance is None:
                raise Exception()
            
            return torch.tensor(closest_label)
        elif any(x == 0 for x in res_shape):
            return torch.zeros(res_shape)
        else:
            max_similarities: torch.Tensor = torch.zeros(*res_shape)
            closest_labels: torch.Tensor = torch.zeros(*res_shape, dtype=torch.int8)
            defined_vars: bool = False
            
            for label, class_hv in self.__classes.items():
                class_hv_2 = class_hv[*((None,) * len(res_shape)), ...].expand(*res_shape, -1, -1, -1)
                similarities: torch.Tensor = functions.normalized_similarity(test_batch, class_hv_2)
                
                if not defined_vars:
                    max_similarities = similarities
                    closest_labels = torch.full(res_shape, label, dtype=torch.int8)
                    defined_vars = True
                    continue
                
                modify_result = similarities > max_similarities
                max_similarities = torch.where(modify_result, similarities, max_similarities)
                closest_labels = torch.where(modify_result, label, closest_labels)
            
            if not defined_vars:
                raise Exception()
            
            return closest_labels

