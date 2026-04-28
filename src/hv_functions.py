import abc
import torch
from math import sqrt

from gpu_management.tensor_functions import TensorFunctionsManager
from constants import D
from get_args import get_arg

_private_key: object = object()

# HVs are represented as torch.Tensor instances of complex numbers, in which the last three dimensions must be depth, row, and column, from first to last
class UpperTensorFunctionsManager:
    lower: TensorFunctionsManager
    
    def __init__(self, lower: TensorFunctionsManager, key: object):
        if key is not _private_key: raise Exception()
        self.lower = lower
    
    # hvs: (x)D batch of HVs
    # returns: (x-1)D batch of HVs
    def sum_hvs(self, hvs: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        tensor = hvs.transpose(-4, -3).transpose(-3, -2).transpose(-2, -1)
        result = self.lower.summation(tensor, 1, out=out)
        return result
    
    # hv1: HV
    # hv2: HV
    # returns: HV
    def sum_two_hvs(self, hv1: torch.Tensor, hv2: torch.Tensor, alpha: float = 1.0, out: torch.Tensor | None = None) -> torch.Tensor:
        return self.lower.weighted_addition(hv1, hv2, alpha=alpha, out=out)
    
    # positional_encodings: (x)D batch of HVs
    # encodings: (x)D batch of HVs
    # returns: (x-1)D batch of HVs
    def query_from_encoded(self, positional_encodings: torch.Tensor, encodings: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        v1 = self.lower.matrix_mult(positional_encodings, encodings)
        del positional_encodings
        del encodings
        
        v2 = self.sum_hvs(v1)
        del v1
        
        v3 = self.lower.normalize(v2, out=out)
        del v2
        
        return v3

    # encodings1: (x)D batch of HVs
    # encodings2: (x)D batch of HVs
    # positions2: (x)D batch of HVs
    # returns: (x-1)D batch of HVs
    # Warning: Modifies the input tensors
    def key_from_encoded(self, encodings1: torch.Tensor, encodings2: torch.Tensor, positions2: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        v1 = self.lower.matrix_mult(encodings2, positions2)
        del encodings2
        del positions2
        
        v2 = self.lower.adjoint(v1)
        del v1
        
        v3 = self.lower.matrix_mult(v2, encodings1)
        del v2
        del encodings1
        
        v4 = self.sum_hvs(v3)
        del v3
        
        v5 = self.lower.normalize(v4, out=out)
        del v4
        
        return v5

    # positional_encodings: (x)D batch of HVs
    # encodings: (x)D batch of HVs
    # returns: (x-1)D batch of HVs
    def value_from_encoded(self, positional_encodings: torch.Tensor, encodings: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        v1 = self.lower.matrix_mult(positional_encodings, encodings)
        del positional_encodings
        del encodings
        
        v2 = self.sum_hvs(v1)
        del v1
        
        v3 = self.lower.normalize(v2, out=out)
        del v2
        
        return v3
    
    def softmax_hv(self, hv: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        tensor = hv.transpose(-3, -1)
        mid_res = self.lower.softmax(tensor)
        result = mid_res.transpose(-3, -1)
        
        if out is not None:
            out[...] = result
            return out
        
        return result
    
    # query_hv: HV
    # key_hv: HV
    # value_hv: HV
    # returns: HV
    def attention_function(self, query_hv: torch.Tensor, key_hv: torch.Tensor, value_hv: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        v1 = self.lower.adjoint(key_hv)
        del key_hv
        
        v2 = self.lower.matrix_mult(query_hv, v1)
        del query_hv
        del v1
        
        v3 = self.lower.real(v2)
        del v2
        
        v4 = self.softmax_hv(v3).type(torch.complex64)
        del v3
        
        v4_5 = self.lower.divide_by_scalar(v4, sqrt(D))
        del v4
        
        v5 = self.lower.matrix_mult(v4_5, value_hv)
        del v4_5
        del value_hv
        
        v6 = self.lower.normalize(v5, out=out)
        del v5
        
        return v6

    # a: (x)D batch of HVs
    # b: (x)D batch of HVs
    # returns: (x)D batch of floating-point numbers from 0 to 1
    def unnormalized_similarity(self, a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError(f"Cannot calculate similarity on tensors of shape {a.shape} and {b.shape}")
        
        shape = a.shape
        if shape[-1] != shape[-2]:
            raise ValueError("Last two dimensions must be identical for both operands")
        
        D = a.shape[-3]
        m = a.shape[-2]

        v1 = self.lower.adjoint(b)
        del b
        
        v2 = self.lower.matrix_mult(a, v1)
        del a
        del v1
        
        v3 = self.lower.summation(v2.transpose(-3, -2).transpose(-2, -1), 1)
        del v2
        
        v4 = self.lower.diagonal(v3)
        del v3
        
        v4_5 = self.lower.summation(v4, 1)
        del v4
        
        v5 = self.lower.real(v4_5)
        del v4_5
        
        v6 = self.lower.divide_by_scalar(v5, m * D, out=out)
        del v5
        
        return v6

    # a: (x)D batch of HVs
    # b: (x)D batch of HVs
    # returns: (x)D batch of floating-point numbers from 0 to 1
    def normalized_similarity(self, a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        mid = self.unnormalized_similarity(a, b)
        v1 = self.unnormalized_similarity(a, a)
        v2 = self.unnormalized_similarity(b, b)
        
        v3 = self.lower.elem_mult(v1, v2)
        del v1
        del v2
        
        v4 = self.lower.sqrt(v3)
        del v3
        
        result = self.lower.elem_div(mid, v4, out=out)
        del mid
        del v4
        
        return torch.nan_to_num(result, out=result)

class Bundling2TensorFunctionsManager(UpperTensorFunctionsManager):
    def sum_hvs(self, hvs: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        tensor = hvs.transpose(-4, -3).transpose(-3, -2).transpose(-2, -1)
        result = torch.zeros(tensor.shape[:-1], dtype=tensor.dtype)
        
        for i in range(tensor.shape[-1]):
            self.lower.addition(result, tensor[..., i], out=result)
            self.lower.normalize(result, out=result)
        
        return result
    
    def sum_two_hvs(self, hv1: torch.Tensor, hv2: torch.Tensor, alpha: float = 1.0, out: torch.Tensor | None = None) -> torch.Tensor:
        result = self.lower.weighted_addition(hv1, hv2, alpha=alpha, out=out)
        self.lower.normalize(result, out=result)
        return result
    
    def query_from_encoded(self, positional_encodings: torch.Tensor, encodings: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        v1 = self.lower.matrix_mult(positional_encodings, encodings)
        del positional_encodings
        del encodings
        
        v2 = self.sum_hvs(v1)
        del v1
        
        return v2
    
    def key_from_encoded(self, encodings1: torch.Tensor, encodings2: torch.Tensor, positions2: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        v1 = self.lower.matrix_mult(encodings2, positions2)
        del encodings2
        del positions2
        
        v2 = self.lower.adjoint(v1)
        del v1
        
        v3 = self.lower.matrix_mult(v2, encodings1)
        del v2
        del encodings1
        
        v4 = self.sum_hvs(v3)
        del v3
        
        return v4
    
    def value_from_encoded(self, positional_encodings: torch.Tensor, encodings: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        v1 = self.lower.matrix_mult(positional_encodings, encodings)
        del positional_encodings
        del encodings
        
        v2 = self.sum_hvs(v1)
        del v1
        
        
        return v2
    
    def softmax_hv(self, hv: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        tensor = hv.transpose(-3, -1)
        mid_res = self.lower.softmax(tensor)
        result = mid_res.transpose(-3, -1)
        
        if out is not None:
            out[...] = result
            return out
        
        return result
    
    def attention_function(self, query_hv: torch.Tensor, key_hv: torch.Tensor, value_hv: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        v1 = self.lower.adjoint(key_hv)
        del key_hv
        
        v2 = self.lower.matrix_mult(query_hv, v1)
        del query_hv
        del v1
        
        v3 = self.lower.real(v2)
        del v2
        
        v4 = self.softmax_hv(v3).type(torch.complex64)
        del v3
        
        v4_5 = self.lower.divide_by_scalar(v4, sqrt(D))
        del v4
        
        v5 = self.lower.matrix_mult(v4_5, value_hv)
        del v4_5
        del value_hv
        
        v6 = self.lower.normalize(v5, out=out)
        del v5
        
        return v6
    
    def unnormalized_similarity(self, a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError(f"Cannot calculate similarity on tensors of shape {a.shape} and {b.shape}")
        
        shape = a.shape
        if shape[-1] != shape[-2]:
            raise ValueError("Last two dimensions must be identical for both operands")
        
        D = a.shape[-3]
        m = a.shape[-2]

        v1 = self.lower.adjoint(b)
        del b
        
        v2 = self.lower.matrix_mult(a, v1)
        del a
        del v1
        
        v3 = self.lower.summation(v2.transpose(-3, -2).transpose(-2, -1), 1)
        del v2
        
        v4 = self.lower.diagonal(v3)
        del v3
        
        v4_5 = self.lower.summation(v4, 1)
        del v4
        
        v5 = self.lower.real(v4_5)
        del v4_5
        
        v6 = self.lower.divide_by_scalar(v5, m * D, out=out)
        del v5
        
        return v6

def get_functions_manager(lower: TensorFunctionsManager) -> UpperTensorFunctionsManager:
    bundling_mode = get_arg("BUNDLING_MODE", "int")
    if bundling_mode == 1:
        return UpperTensorFunctionsManager(lower, _private_key)
    elif bundling_mode == 2:
        return Bundling2TensorFunctionsManager(lower, _private_key)
    else:
        raise Exception(f"Unknown bundling mode: {bundling_mode}")

__all__ = ["UpperTensorFunctionsManager", "get_functions_manager"]
