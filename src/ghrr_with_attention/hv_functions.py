import torch
import torch.nn.functional as F
from ghrr_with_attention.utils import value_or
from typing import overload
from functools import reduce

# HVs are represented as torch.Tensor instances of complex numbers, in which the last three dimensions must be depth, row, and column, from first to last

# data: Batch of HVs
# returns: Batch of HVs
def normalize(data: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    data_size = reduce(lambda a, b: a * b, data.shape[-3:])
    view = data.view(-1, data_size)
    return F.normalize(data, out=out)

def add_grouped(data: torch.Tensor, *, dim: tuple[int, ...] | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    dim_: tuple[int, ...] = value_or(dim, tuple(range(len(data.shape) - 3)))

    for n in range(1, 4):
        dim_id = len(data.shape) - n
        if dim_id in dim_:
            raise ValueError(F"Dimension {dim_id} is internal to the structure of HVs")
    
    return torch.sum(data, dim=dim_, out=out)

def mult(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    return torch.matmul(a, b, out=out)

def bundle_grouped(data: torch.Tensor, *, dim: tuple[int, ...] | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    v1 = add_grouped(data, dim=dim, out=out)
    normalize(v1, out=v1)
    return v1

def bind(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    v1 = mult(a, b, out=out)
    normalize(v1, out=v1)
    return v1
