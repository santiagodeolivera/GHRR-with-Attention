from pathlib import Path
from typing import Callable, Any

import torch

from tensor_proxy import TensorProxy
from memory import MemoryManager
from utils import get_size
from constants import element_type

class TensorFunctionsManager:
    __manager: MemoryManager
    use_gpu: bool
    
    def __init__(self, manager: MemoryManager, use_gpu: bool = True):
        self.__manager = manager
        self.use_gpu = use_gpu
    
    def final_use_gpu(self, spec: bool | None) -> bool:
        if spec is None: return self.__use_gpu
        return spec
    
    def randn(self, shape: tuple[int, ...], out: Path) -> TensorProxy:
        result = TensorProxy.empty(shape, out)
        tensor = result.tensor()
        tensor[...] = torch.randn(shape, dtype=element_type, device="cpu")
        return result
    
    def element_wise_unary_operation(self, \
        v1: TensorProxy, \
        fn: Callable[[torch.Tensor, torch.Tensor], Any], \
        out: Path
    ) -> TensorProxy:
        size = get_size(v1.shape)
        max_size = self.__manager.max_mem // 3
        if max_size <= 0: raise Exception()
        
        t1 = v1.tensor().view(-1)
        result = TensorProxy.empty(v1.shape, out)
        t2 = result.tensor().view(-1)
        
        if self.use_gpu:
            g1, g2 = self.__manager.alloc_tensors((max_size,) for _ in range(2))
            
            start = 0
            while start < size:
                stop = min(size, start + max_size)
                g01 = g1[:stop - start]
                g02 = g2[:stop - start]
                
                g01[...] = t1[start:stop]
                
                fn(g01, g02)
                
                t2[start:stop] = g02
                start = stop
        else:
            fn(t1, t2)
        
        return result
    
    def element_wise_binary_operation(self, \
        v1: TensorProxy, \
        v2: TensorProxy, \
        fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any], \
        out: Path \
    ) -> TensorProxy:
        if v1.shape != v2.shape:
            raise ValueError("Elements must have the same shape")
        
        size = get_size(v1.shape)
        max_size = self.__manager.max_mem // 3
        if max_size <= 0: raise Exception()
        
        t1 = v1.tensor().view(-1)
        t2 = v2.tensor().view(-1)
        result = TensorProxy.empty(v1.shape, out)
        t3 = result.tensor().view(-1)
        
        if self.use_gpu:
            g1, g2, g3 = self.__manager.alloc_tensors((max_size,) for _ in range(3))
            
            start = 0
            while start < size:
                stop = min(size, start + max_size)
                g01 = g1[:stop - start]
                g02 = g2[:stop - start]
                g03 = g3[:stop - start]
                
                g01[...] = t1[start:stop]
                g02[...] = t2[start:stop]
                
                fn(g01, g02, g03)
                
                t3[start:stop] = g03
                start = stop
        else:
            fn(t1, t2, t3)
        
        return result

    def addition(self, v1: TensorProxy, v2: TensorProxy, out: Path) -> TensorProxy:
        return self.element_wise_binary_operation(v1, v2, lambda t1, t2, t3: torch.add(t1, t2, out=t3), out)
    
    def matrix_mult(self, v1: TensorProxy, v2: TensorProxy, out: Path) -> TensorProxy:
        shape1 = v1.shape
        shape2 = v2.shape
        
        if shape1[-1] != shape2[-2]:
            raise ValueError("Elements aren't compatible for matrix multiplication")
        
        if shape1[:-2] != shape2[:-2]:
            raise ValueError("Incompatible batch dimensions")
        
        rows = shape1[-2]
        n = shape1[-1]
        cols = shape2[-1]
        
        num_matrices = get_size(shape1[:-2])
        mem_per_op = rows * n + n * cols + rows * cols
        max_parallel_operations = self.__manager.max_mem // mem_per_op
        if max_parallel_operations <= 0: raise Exception()
        
        t1 = v1.tensor().view(-1, rows, n)
        t2 = v2.tensor().view(-1, n, cols)
        result = TensorProxy.empty((*shape1[:-2], rows, cols), out)
        t3 = result.tensor().view(-1, rows, cols)
        
        g1, g2, g3 = self.__manager.alloc_tensors( \
            (max_parallel_operations, a, b) for a, b in ((rows, n), (n, cols), (rows, cols)) \
        )
        
        start = 0
        while start < num_matrices:
            stop = min(num_matrices, start + max_parallel_operations)
            g01 = g1[:stop - start]
            g02 = g2[:stop - start]
            g03 = g3[:stop - start]
            
            g01[...] = t1[start:stop]
            g02[...] = t2[start:stop]
            
            torch.matmul(g01, g02, out=g03)
            
            t3[start:stop] = g03
            start = stop
    
        return result
    
    def get_norm(self, v1: TensorProxy) -> float:
        return torch.linalg.vector_norm(v1.tensor())
    
    def divide_by_scalar(self, v1: TensorProxy, divisor: float, out: Path) -> TensorProxy:
        return self.element_wise_unary_operation(v1, lambda t1, t2: torch.div(t1, divisor, out=t2), out)
    
    def normalize(self, v1: TensorProxy, out: Path) -> TensorProxy:
        norm = self.get_norm(v1)
        return self.divide_by_scalar(v1, norm, out=out)
    
    def adjoint(self, v1: TensorProxy, out: Path) -> TensorProxy:
        shape = v1.shape
        rows = shape[-2]
        cols = shape[-1]
        
        t1 = v1.tensor().view(-1, rows, cols)
        result = TensorProxy.empty((*shape[:-2], cols, rows), out)
        t2 = result.tensor().view(-1, cols, rows)
        t2[...] = torch.adjoint(t1)
        
        return result
    
    def real(self, v1: TensorProxy, out: Path) -> TensorProxy:
        t1 = v1.tensor()
        result = TensorProxy.empty(v1.shape, out)
        t2 = result.tensor()
        
        t2[...] = t1.real.type(element_type)
        
        return result
    
