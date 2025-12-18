import torch
import unittest
from math import nan

def torch_singular_value_ind(data: torch.Tensor, *, data_dims: int) -> torch.Tensor:
    dims = len(data.shape)
    start_dim = dims - data_dims
    
    flat = data.flatten(start_dim=start_dim)
    
    slice = flat[..., 0:1]
    
    return (                                       \
        (                                          \
            flat == slice                          \
        ) | (                                      \
            torch.isnan(flat) & torch.isnan(slice) \
        )                                          \
    )

def torch_singular_value(data: torch.Tensor, *, data_dims: int) -> bool:
    return torch_singular_value_ind(data, data_dims=data_dims).all()
