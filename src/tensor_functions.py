from pathlib import Path
from functools import reduce

from tensor_proxy import TensorProxy
from memory import MemoryManager

"""
def addition(v1: TensorProxy, v2: TensorProxy, out: Path) -> TensorProxy:
    if v1.shape != v2.shape:
        raise ValueError("Elements must have the same shape")
    
    size = reduce(lambda a, b: a * b, v1.shape, 1)
    
    memory = MemoryManager.get()
    if memory.max_mem < size * 2
"""

