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
    
    def new_names(self, n: int) -> tuple[Path, ...]:
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
    
    # positional_encodings: (x)D batch of HVs
    # encodings: (x)D batch of HVs
    # returns: (x-1)D batch of HVs
    def query_from_encoded(self, positional_encodings: torch.Tensor, encodings: torch.Tensor, out: Path) -> TensorProxy:
        v1 = mult(positional_encodings, encodings)
        v2 = add_grouped(v1, dim=-4)
        v3 = normalize(v2)
        return v3

    # encodings1: (x)D batch of HVs
    # encodings2: (x)D batch of HVs
    # positions2: (x)D batch of HVs
    # returns: (x-1)D batch of HVs
    # Warning: Modifies the input tensors
    def key_from_encoded(self, encodings1: torch.Tensor, encodings2: torch.Tensor, positions2: torch.Tensor) -> TensorProxy:
        v1 = mult_batched(encodings2, positions2, out=encodings2, batch_size=256)
        v2 = v1.adjoint()
        v3 = mult_batched(v2, encodings1, out=encodings1, batch_size=256)
        v4 = add_grouped(v3, dim=-4)
        v5 = normalize(v4)
        return v5

    # positional_encodings: (x)D batch of HVs
    # encodings: (x)D batch of HVs
    # returns: (x-1)D batch of HVs
    def value_from_encoded(self, positional_encodings: torch.Tensor, encodings: torch.Tensor) -> TensorProxy:
        v1 = mult(positional_encodings, encodings)
        v2 = add_grouped(v1, dim=-4)
        v3 = normalize(v2)
        return v3

    # query_hv: HV
    # key_hv: HV
    # value_hv: HV
    # returns: HV
    def attention_function(self, query_hv: torch.Tensor, key_hv: torch.Tensor, value_hv: torch.Tensor) -> TensorProxy:
        v1 = torch.adjoint(key_hv)
        v2 = mult(query_hv, v1)
        v3 = v2.real
        v4 = torch.nn.functional.softmax(v3, dim=-3).type(torch.complex64)
        v5 = mult(v4, value_hv)
        v6 = normalize(v5)
        return v6

    # a: (x)D batch of HVs
    # b: (x)D batch of HVs
    # returns: (x)D batch of floating-point numbers from 0 to 1
    def unnormalized_similarity(self, a: torch.Tensor, b: torch.Tensor, *, batch_size: int | None = None) -> TensorProxy:
        if a.shape != b.shape:
            raise ValueError(f"Cannot calculate similarity on tensors of shape {a.shape} and {b.shape}")
        
        shape = a.shape
        if shape[-1] != shape[-2]:
            raise ValueError("Last two dimensions must be identical for both operands")
        
        D = a.shape[-3]
        m = a.shape[-2]

        if batch_size is None:
            v1 = torch.adjoint(b)
            v2 = mult(a, v1)
            v3 = torch.sum(v2, dim=-3)
            v4 = torch.diagonal(v3, dim1=-2, dim2=-1).sum(dim=-1)
            v5 = torch.real(v4)
            v6 = v5 / (m * D)
            return v6
        else:
            res_shape = shape[:-3]
            a_view = a.view(-1, D, m, m)
            b_view = b.view(-1, D, m, m)
            num_hvs = a_view.shape[0]
            
            res = torch.zeros(size=res_shape, device=default_device, dtype=localTypes.hvRealType)
            res_view = res.view(-1)
            
            for i in range(0, num_hvs, batch_size):
                range_min = i
                range_max = min(i + batch_size, num_hvs)
                
                res_view[range_min:range_max] = unnormalized_similarity( \
                    a_view[range_min:range_max, ...], \
                    b_view[range_min:range_max, ...] \
                )
            
            return res

    # a: (x)D batch of HVs
    # b: (x)D batch of HVs
    # returns: (x)D batch of floating-point numbers from 0 to 1
    def normalized_similarity(self, a: torch.Tensor, b: torch.Tensor, *, batch_size: int | None = None) -> TensorProxy:
        mid = unnormalized_similarity(a, b, batch_size=batch_size)
        v1 = unnormalized_similarity(a, a, batch_size=batch_size)
        v2 = unnormalized_similarity(b, b, batch_size=batch_size)
        
        return mid / torch.sqrt(v1 * v2)
    
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
    
