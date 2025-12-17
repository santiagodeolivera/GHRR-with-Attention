import torch
import unittest

def torch_singular_value(data: torch.Tensor, *, data_dims: int) -> bool:
    dims = len(data.shape)
    start_dim = dims - data_dims
    
    flat = data.flatten(start_dim=start_dim)
    
    return (flat == flat[..., 0:1]).all()
