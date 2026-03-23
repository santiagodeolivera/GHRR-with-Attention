import torch
import torch.nn.functional as F
from typing import Callable, Any
from functools import reduce
from pathlib import Path

from utils import value_or, not_none, print_tensor_struct
import localTypes
from gpu_management import TensorFunctionsManager

# HVs are represented as torch.Tensor instances of complex numbers, in which the last three dimensions must be depth, row, and column, from first to last

class TmpGenerator:
    __counter: int
    __function: Callable[[int], Path]
    
    def __init__(self, function: Callable[[int], Path]):
        self.__counter = 0
        self.__function = function
    
    def new_paths(self, n: int) -> tuple[Path, ...]:
        start = self.__counter
        stop = start + n
        
        result = tuple((self.__function)(x) for x in range(start, stop))
        
        self.__counter = stop
        return result

class UpperTensorFunctionsManager:
    lower: TensorFunctionsManager
    __tmp_gen: TmpGenerator
    
    def __init__(self, lower: TensorFunctionsManager, function: Callable[[int], Path]):
        self.lower = lower
        self.__tmp_gen = TmpGenerator(function)
    
    # hvs: (x)D batch of HVs
    # returns: (x-1)D batch of HVs
    def sum_hvs(self, hvs: torch.Tensor, *, out: Path) -> TensorProxy:
        tensor = hvs.transpose(-4, -3).transpose(-3, -2).transpose(-2, -1)
        result = self.lower.summation(tensor, 1, out=out)
        return result
    
    # positional_encodings: (x)D batch of HVs
    # encodings: (x)D batch of HVs
    # returns: (x-1)D batch of HVs
    def query_from_encoded(self, positional_encodings: torch.Tensor, encodings: torch.Tensor, *, out: Path) -> TensorProxy:
        tmp = self.__tmp_gen.new_paths(2)
        
        v1 = self.lower.matrix_mult(positional_encodings, encodings, out=tmp[0])
        v2 = self.sum_hvs(v1.tensor(), out=tmp[1])
        v3 = self.lower.normalize(v2.tensor(), out=out)
        return v3

    # encodings1: (x)D batch of HVs
    # encodings2: (x)D batch of HVs
    # positions2: (x)D batch of HVs
    # returns: (x-1)D batch of HVs
    # Warning: Modifies the input tensors
    def key_from_encoded(self, encodings1: torch.Tensor, encodings2: torch.Tensor, positions2: torch.Tensor, *, out: Path) -> TensorProxy:
        tmp = self.__tmp_gen.new_paths(4)
        
        v1 = self.lower.matrix_mult(encodings2, positions2, out=tmp[0])
        v2 = self.lower.adjoint(v1.tensor(), out=tmp[1])
        v3 = self.lower.matrix_mult(v2.tensor(), encodings1, out=tmp[2])
        v4 = self.sum_hvs(v3.tensor(), out=tmp[3])
        v5 = self.lower.normalize(v4.tensor(), out=out)
        return v5

    # positional_encodings: (x)D batch of HVs
    # encodings: (x)D batch of HVs
    # returns: (x-1)D batch of HVs
    def value_from_encoded(self, positional_encodings: torch.Tensor, encodings: torch.Tensor, *, out: Path) -> TensorProxy:
        tmp = self.__tmp_gen.new_paths(2)
        
        v1 = self.lower.matrix_mult(positional_encodings, encodings, out=tmp[0])
        v2 = self.sum_hvs(v1.tensor(), out=tmp[1])
        v3 = self.lower.normalize(v2.tensor(), out=out)
        return v3
    
    def softmax_hv(hv: torch.Tensor, *, out: Path) -> torch.Tensor:
        tensor = hv.tensor().transpose(-3, -1)
        result = self.lower.softmax(tensor, out=out)
        return result.tensor().transpose(-3, -1)
    
    # query_hv: HV
    # key_hv: HV
    # value_hv: HV
    # returns: HV
    def attention_function(self, query_hv: torch.Tensor, key_hv: torch.Tensor, value_hv: torch.Tensor, *, out: Path) -> TensorProxy:
        tmp = self.__tmp_gen.new_paths(5)
        
        v1 = self.lower.adjoint(key_hv, out=tmp[0])
        v2 = self.lower.matrix_mult(query_hv, v1.tensor(), out=tmp[1])
        v3 = self.lower.real(v2.tensor(), out=tmp[2])
        t4 = self.lower.softmax(v3.tensor(), out=tmp[3])
        v5 = self.lower.matrix_mult(t4, value_hv, out=tmp[4])
        v6 = self.lower.normalize(v5.tensor(), out=out)
        return v6

    # a: (x)D batch of HVs
    # b: (x)D batch of HVs
    # returns: (x)D batch of floating-point numbers from 0 to 1
    def unnormalized_similarity(self, a: torch.Tensor, b: torch.Tensor, *, out: Path) -> TensorProxy:
        if a.shape != b.shape:
            raise ValueError(f"Cannot calculate similarity on tensors of shape {a.shape} and {b.shape}")
        
        shape = a.shape
        if shape[-1] != shape[-2]:
            raise ValueError("Last two dimensions must be identical for both operands")
        
        D = a.shape[-3]
        m = a.shape[-2]

        tmp = self.__tmp_gen.new_paths(6)
        
        v1 = self.lower.adjoint(b, out=tmp[0])
        v2 = self.lower.matrix_mult(a, v1.tensor(), out=tmp[1])
        v3 = self.lower.summation(v2.tensor().transpose(-3, -2).transpose(-2, -1), out=tmp[2])
        v4 = self.lower.diagonal(v3.tensor(), out=tmp[3])
        v4_5 = self.lower.summation(v4.tensor(), out=tmp[4])
        v5 = self.lower.real(v4_5.tensor(), out=tmp[5])
        v6 = self.lower.divide_by_scalar(v5.tensor(), m * D, out=out)
        
        return v6

    # a: (x)D batch of HVs
    # b: (x)D batch of HVs
    # returns: (x)D batch of floating-point numbers from 0 to 1
    def normalized_similarity(self, a: torch.Tensor, b: torch.Tensor, *, out: Path) -> TensorProxy:
        tmp = self.__tmp_gen.new_paths(3)
        
        mid = self.unnormalized_similarity(a, b, out=tmp[0])
        v1 = self.unnormalized_similarity(a, a, out=tmp[1])
        v2 = self.unnormalized_similarity(b, b, out=tmp[2])
        
        v3 = self.lower.matrix_mult(v1.tensor(), v2.tensor(), out=tmp[3])
        v4 = self.lower.sqrt(v3.tensor(), out=tmp[4])
        result = self.lower.div(mid.tensor(), v4.tensor(), out=out)
        return result
    
    """
    def torch_fromfunction( \
        fn: Callable[..., torch.Tensor], \
        shape: tuple[int, ...], \
        *, \
        device: torch.device \
    ) -> Any:
        tensors1 = tuple(torch.tensor(range(n), dtype=torch.int32, device=device) for n in shape)
        tensors2 = torch.meshgrid(*tensors1, indexing="ij")
        
        res = fn(*tensors2)
        
        return res
    """
    
