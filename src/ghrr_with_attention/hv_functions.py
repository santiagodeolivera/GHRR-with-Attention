import torch
import torch.nn.functional as F
from ghrr_with_attention.utils import value_or, not_none, print_tensor_struct
from typing import Callable
from functools import reduce

# HVs are represented as torch.Tensor instances of complex numbers, in which the last three dimensions must be depth, row, and column, from first to last

def torch_fromfunction(              \
    fn: Callable[..., torch.Tensor], \
    shape: tuple[int, ...],          \
    *,                               \
    device: torch.device,            \
    dtype: torch.dtype | None = None \
) -> torch.Tensor:
    tensors1 = tuple(torch.tensor(range(n), dtype=torch.int32, device=device) for n in shape)
    tensors2 = torch.meshgrid(*tensors1, indexing="ij")
    
    res = fn(*tensors2)
    if dtype is not None:
        res = res.to(dtype)
    
    return res

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
def add_grouped(data: torch.Tensor, *, dim: tuple[int, ...] | int | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
    if dim is None: dim = tuple(range(len(data.shape) - 3))
    if type(dim) == int: dim = (dim,)

    for n in range(1, 4):
        dim_id = len(data.shape) - n
        if dim_id in dim or -n in dim:
            raise ValueError(F"Dimension {dim_id} (-{n}) is internal to the structure of HVs")
    
    return torch.sum(data, dim=dim, out=out)

# a: (x)D batch of HVs
# b: (x)D batch of HVs
# returns: (x)D batch of HVs
def mult(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    try:
        return torch.matmul(a, b, out=out)
    except Exception as e:
        v1 = print_tensor_struct(out) if out is not None else None
        raise Exception(f"Exception occurred ({print_tensor_struct(a)} * {print_tensor_struct(b)}, out={v1})") from e

# positional_encodings: (x)D batch of HVs
# encodings: (x)D batch of HVs
# returns: (x-1)D batch of HVs
def query_from_encoded(positional_encodings: torch.Tensor, encodings: torch.Tensor) -> torch.Tensor:
    v1 = mult(positional_encodings, encodings)
    v2 = add_grouped(v1, dim=-4)
    v3 = normalize(v2)
    return v3

# encodings1: (x)D batch of HVs
# encodings2: (x)D batch of HVs
# positions2: (x)D batch of HVs
# returns: (x-1)D batch of HVs
def key_from_encoded(encodings1: torch.Tensor, encodings2: torch.Tensor, positions2: torch.Tensor) -> torch.Tensor:
    v1 = mult(encodings2, positions2)
    v2 = v1.adjoint()
    v3 = mult(v2, encodings1)
    v4 = add_grouped(v3, dim=-4)
    v5 = normalize(v4)
    return v5

# positional_encodings: (x)D batch of HVs
# encodings: (x)D batch of HVs
# returns: (x-1)D batch of HVs
def value_from_encoded(positional_encodings: torch.Tensor, encodings: torch.Tensor) -> torch.Tensor:
    v1 = mult(positional_encodings, encodings)
    v2 = add_grouped(v1, dim=-4)
    v3 = normalize(v2)
    return v3

# a: (x)D batch of HVs
# b: (x)D batch of HVs
# returns: (x)D batch of floating-point numbers from 0 to 1
def unnormalized_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Cannot calculate similarity on tensors of shape {a.shape} and {b.shape}")
    
    shape = a.shape
    if shape[-1] != shape[-2]:
        raise ValueError("Last two dimensions must be identical for both operands")
    
    D = a.shape[-3]
    m = a.shape[-2]
    
    v1 = b.adjoint()
    v2 = torch.matmul(a, v1)
    v3 = torch.sum(v2, dim=-3)
    v4 = torch.diagonal(v3, dim1=-2, dim2=-1).sum(dim=-1)
    v5 = torch.real(v4)
    v6 = v5 / (m * D)
    return v6

# a: (x)D batch of HVs
# b: (x)D batch of HVs
# returns: (x)D batch of floating-point numbers from 0 to 1
def normalized_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mid = unnormalized_similarity(a, b)
    v1 = unnormalized_similarity(a, a)
    v2 = unnormalized_similarity(b, b)
    
    return mid / torch.sqrt(v1 * v2)
