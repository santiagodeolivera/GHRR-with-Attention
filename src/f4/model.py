from pathlib import Path
from typing import Iterable
from dataclasses import dataclass

import torch

from constants import D, m
from hv_proxy import HVProxy
from hv_functions import UpperTensorFunctionsManager
from gpu_management.data_type import DataType
from mmap_tensors import MmapTensors
from utils import get_size, clamp, AccAvg
from time_ import Timer

@dataclass
class PredictionData:
    label: int
    top1: float
    top2: float
    
    @property
    def label_sim(self) -> float:
        return self.top1

def get_class_hv_from_file(path: Path) -> tuple[int, torch.Tensor]:
    id = int(path.stem)
    hv = MmapTensors.read_if_exists(path)
    if hv is None:
        raise Exception()
    return (id, hv)

def new_class_hv(root: Path, id: int) -> tuple[int, torch.Tensor]:
    hv = MmapTensors.new_override(root / f"{id}", (D, m, m), DataType.complex64)
    if hv is None:
        raise Exception()
    return (id, hv)

def clone_hv(src: torch.Tensor, path: Path) -> torch.Tensor:
    hv = MmapTensors.new_override(path, (D, m, m), DataType.complex64)
    if hv is None:
        raise Exception()
    
    hv[...] = src
    return hv

match get_arg("BUNDLING_MODE", "int"):
    case 1:
    def normalize_and_similarity(functions: UpperTensorFunctionsManager, \
        hv: torch.Tensor, class_hv: torch.Tensor, normalized_hv_space: torch.Tensor \
    ) -> float:
        normalized_class_hv = functions.lower.normalize(class_hv, out=normalized_hv_space)
        return functions.normalized_similarity(hv, normalized_class_hv).item()
    
    case 2:
    def normalize_and_similarity(functions: UpperTensorFunctionsManager, \
        hv: torch.Tensor, class_hv: torch.Tensor, normalized_hv_space: torch.Tensor \
    ) -> float:
        return functions.normalized_similarity(hv, class_hv).item()
    
    case _:
    raise Exception()

# TODO: Refactor similar functions
class Model:
    __classes: dict[int, torch.Tensor]
    
    def __init__(self, classes: dict[int, torch.Tensor]):
        self.__classes = classes
    
    @staticmethod
    def new_empty(root_dir: Path, classes: Iterable[int]) -> "Model":
        class_hv_iter = ( new_class_hv(root_dir, label) for label in classes )
        class_hvs: dict[int, torch.Tensor] = dict(class_hv_iter)
        return Model(class_hvs)
    
    def clone(self, root_dir: Path) -> "Model":
        class_hv_iter = ( (label, clone_hv(hv, root_dir / f"{label}")) for label, hv in self.__classes.items() )
        class_hvs: dict[int, torch.Tensor] = dict(class_hv_iter)
        return Model(class_hvs)
    
    @staticmethod
    def from_fs(root_dir: Path) -> "Model":
        class_hv_iter = ( get_class_hv_from_file(path) for path in root_dir.glob("*.json") )
        classes: dict[int, torch.Tensor] = dict(class_hv_iter)
        return Model(classes)
    
    def train(self, functions: UpperTensorFunctionsManager, training_set: Iterable[HVProxy], *, msg: str | None = None) -> None:
        error_calculator = AccAvg(default = 1.0)
        
        norm_class_hv_space = torch.empty((D, m, m), dtype=torch.complex64)
        new_hv_space = torch.empty((D, m, m), dtype=torch.complex64)
        for i, proxy in enumerate(training_set):
            timer = Timer(f"{msg}: HV {i}" if msg is not None else f"HV {i}")
            
            label = proxy.label
            class_hv = self.__classes[label]
            new_hv = proxy.get_hv(out=new_hv_space)
            
            prediction: PredictionData = self.__test_single(functions, new_hv)
            similarity_to_label = normalize_and_similarity(functions, new_hv, class_hv, norm_class_hv_space)
            w: float = clamp(1 - prediction.top1 - prediction.top2, 0.0, 1.0)
            if prediction.label != label:
                functions.sum_two_hvs(class_hv, new_hv, alpha=(1 - similarity_to_label) * w, out=class_hv)
                
                pred_class_hv = self.__classes[prediction.label]
                functions.sum_two_hvs(pred_class_hv, new_hv, alpha=(prediction.label_sim - 1) * w, out=pred_class_hv)
                
                error_calculator.add(similarity_to_label)
            elif prediction.label_sim <= error_calculator.get():
                functions.sum_two_hvs(class_hv, new_hv, alpha=w, out=class_hv)
            
            timer.end()
        
        for class_hv in self.__classes.values():
            functions.lower.normalize(class_hv, out=class_hv)
    
    def __test_single(self, functions: UpperTensorFunctionsManager, hv: torch.Tensor) -> PredictionData:
        min_distance: float | None = None
        second_min_distance: float | None = None
        closest_label: int = -1
        
        norm_class_hv_space = torch.empty((D, m, m), dtype=torch.complex64)
        distances: Iterable[tuple[int, float]] = tuple( \
            (label, normalize_and_similarity(functions, hv, class_hv, norm_class_hv_space)) \
            for label, class_hv in self.__classes.items() \
        )
        
        closest_label, top1 = max(distances, key=lambda v: v[1])
        
        distances_excluding_first = ((label, class_hv) for label, class_hv in distances if label != closest_label)
        _1, top2 = max(distances_excluding_first, key=lambda v: v[1])
        
        return PredictionData(label=closest_label, top1=top1, top2=top2)

