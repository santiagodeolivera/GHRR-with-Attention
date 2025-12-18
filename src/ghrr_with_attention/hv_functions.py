import torch
import torch.nn.functional as F
from ghrr_with_attention.utils import value_or, not_none
from typing import overload
from functools import reduce

# HVs are represented as torch.Tensor instances of complex numbers, in which the last three dimensions must be depth, row, and column, from first to last

# data: (x)D batch of HVs
# returns: (x)D batch of HVs
def normalize(data: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    data_size = reduce(lambda a, b: a * b, data.shape[-3:])
    view = data.view(-1, data_size)
    
    real_out: torch.Tensor
    if not_none(out):
        real_out = out
    else:
        real_out = torch.empty_like(data)
    real_out_view = real_out.view(-1, data_size)
    
    F.normalize(view, p=2, dim=1, out=real_out_view)
    return real_out

# data: (x)D batch of HVs
# dims: Dimensions to sum
# returns: (x-dims.len)D batch of HVs
def add_grouped(data: torch.Tensor, *, dim: tuple[int, ...] | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    dim_: tuple[int, ...] = value_or(dim, tuple(range(len(data.shape) - 3)))

    for n in range(1, 4):
        dim_id = len(data.shape) - n
        if dim_id in dim_:
            raise ValueError(F"Dimension {dim_id} is internal to the structure of HVs")
    
    return torch.sum(data, dim=dim_, out=out)

# a: (x)D batch of HVs
# b: (x)D batch of HVs
# returns: (x)D batch of HVs
def mult(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    return torch.matmul(a, b, out=out)

# data: (x)D batch of HVs
# dims: Dimensions to sum
# returns: (x-dims.len)D batch of HVs
def bundle_grouped(data: torch.Tensor, *, dim: tuple[int, ...] | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    v1 = add_grouped(data, dim=dim, out=out)
    normalize(v1, out=v1)
    return v1

# a: (x)D batch of HVs
# b: (x)D batch of HVs
# returns: (x)D batch of HVs
def bind(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    v1 = mult(a, b, out=out)
    normalize(v1, out=v1)
    return v1

def query_from_encoded(positional_encodings: torch.Tensor, encodings: torch.Tensor) -> torch.Tensor:
    v1 = mult(positional_encodings, encodings)
    v2 = add_grouped(v1)
    return normalize(v2)

def key_from_encoded(encodings1: torch.Tensor, encodings2: torch.Tensor, positions2: torch.Tensor) -> torch.Tensor:
    v1 = mult(encodings2, positions2)
    v2 = v1.adjoint()
    v3 = mult(v2, encodings1)
    return normalize(v3)

def value_from_encoded(positional_encodings: torch.Tensor, encodings: torch.Tensor) -> torch.Tensor:
    v1 = mult(positional_encodings, encodings)
    v2 = add_grouped(v1)
    return normalize(v2)
