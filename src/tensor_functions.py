from pathlib import Path
from contextlib import ExitStack
from typing import Callable, Any

import torch

from tensor_proxy import TensorProxy
from memory import MemoryManager
from utils import get_size
from constants import element_type

def randn(shape: tuple[int, ...], out: Path) -> TensorProxy:
    result = TensorProxy.empty(shape, out)
    tensor = result.tensor()
    tensor[...] = torch.randn(shape, dtype=element_type, device="cpu")
    return result

def element_wise_binary_operation( \
    v1: TensorProxy, \
    v2: TensorProxy, \
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any], \
    out: Path
) -> TensorProxy:
    if v1.shape != v2.shape:
        raise ValueError("Elements must have the same shape")
    
    size = get_size(v1.shape)
    
    manager = MemoryManager.get()
    max_size = manager.max_mem // 3
    if max_size <= 0: raise Exception()
    
    t1 = v1.tensor()
    t2 = v2.tensor()
    result = TensorProxy.empty(v1.shape, out)
    t3 = result.tensor()
    
    view1 = t1.view(-1)
    view2 = t2.view(-1)
    view3 = t3.view(-1)
    
    start = 0
    with ExitStack() as stack:
        g1, g2, g3 = tuple( \
            stack.enter_context(manager.alloc((max_size,))) \
            for _ in range(3) \
        )
        
        while start < size:
            stop = min(size, start + max_size)
            g1.tensor[...] = view1[start:stop]
            g2.tensor[...] = view2[start:stop]
            
            fn(g1.tensor, g2.tensor, g3.tensor)
            
            view3[start:stop] = g3.tensor
            start = stop
    
    return result

def addition(v1: TensorProxy, v2: TensorProxy, out: Path) -> TensorProxy:
    return element_wise_binary_operation(v1, v2, lambda t1, t2, t3: torch.add(t1, t2, out=t3), out)

def matrix_mult(v1: TensorProxy, v2: TensorProxy, out: Path) -> TensorProxy:
    s1 = v1.shape
    s2 = v2.shape
    
    if s1[-2] != s2[-1]:
        raise ValueError("Elements aren't compatible for matrix multiplication")
    
    if s1[:-2] != s2[:-2]:
        raise ValueError("Incompatible batch dimensions")
    
    rows = s1[-1]
    n = s2[-1]
    cols = s2[-2]
    
    
