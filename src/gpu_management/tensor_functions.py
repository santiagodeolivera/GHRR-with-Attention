from pathlib import Path
from typing import Callable, Any
from builtins import slice

import torch

from .tensor_proxy import TensorProxy
from .memory import MemoryManager
from .data_type import DataType
from utils import get_size

class TensorFunctionsManager:
    __managers: dict[str, MemoryManager]
    use_gpu: bool # Do not touch
    
    def __init__(self, manager_mem: dict[str, int]):
        self.__managers = {k: MemoryManager.create(v, DataType.get_by_name(k)) for k, v in manager_mem.items()}
        self.use_gpu = True
    
    def __enter__(self) -> "TensorFunctionsManager":
        return self
    
    def __exit__(self, exc_t, exc_v, exc_tb) -> None:
        for manager in self.__managers.values():
            manager.__exit__(exc_t, exc_v, exc_tb)
    
    def randn(self, shape: tuple[int, ...], out: Path, data_type: DataType) -> TensorProxy:
        result = TensorProxy.empty_override(shape, out, data_type)
        tensor = result.tensor()
        torch.randn(shape, dtype=data_type.value, out=tensor)
        return result
    
    def element_wise_unary_operation(self, \
        v1: torch.Tensor, \
        fn: Callable[[torch.Tensor, torch.Tensor], Any], \
        out: Path
    ) -> TensorProxy:
        data_type = DataType.get_by_dtype(v1.dtype)
        manager = self.__managers[data_type.name]
        
        size = get_size(v1.shape)
        max_size = manager.max_mem // 3
        if max_size <= 0: raise Exception()
        
        t1 = v1.view(-1)
        result = TensorProxy.empty_override(v1.shape, out, data_type)
        t2 = result.tensor().view(-1)
        
        if self.use_gpu:
            g1, g2 = manager.alloc_tensors((max_size,) for _ in range(2))
            
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
        v1: torch.Tensor, \
        v2: torch.Tensor, \
        fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any], \
        out: Path \
    ) -> TensorProxy:
        if v1.dtype != v2.dtype:
            raise Exception()
        
        data_type = DataType.get_by_dtype(v1.dtype)
        manager = self.__managers[data_type.name]
        
        if v1.shape != v2.shape:
            raise ValueError("Elements must have the same shape")
        
        size = get_size(v1.shape)
        max_size = manager.max_mem // 3
        if max_size <= 0: raise Exception()
        
        t1 = v1.view(-1)
        t2 = v2.view(-1)
        result = TensorProxy.empty_override(v1.shape, out, data_type)
        t3 = result.tensor().view(-1)
        
        if self.use_gpu:
            g1, g2, g3 = manager.alloc_tensors((max_size,) for _ in range(3))
            
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

    def addition(self, v1: torch.Tensor, v2: torch.Tensor, *, out: Path) -> TensorProxy:
        return self.element_wise_binary_operation(v1, v2, lambda t1, t2, t3: torch.add(t1, t2, out=t3), out)
    
    def summation(self, v1: torch.Tensor, unit_dims: int, *, out: Path) -> TensorProxy:
        data_type = DataType.get_by_dtype(v1.dtype)
        
        if unit_dims <= 0:
            raise Exception()
        
        shape = v1.shape
        if len(shape) < unit_dims:
            raise Exception()
        
        batch_shape = shape[:-unit_dims]
        unit_shape = shape[-unit_dims:]
        
        batch_size = get_size(batch_shape)
        unit_size = get_size(unit_shape)
        
        t1 = v1.view(-1, unit_size)
        result = TensorProxy.empty_override(batch_shape, out, data_type)
        t2 = result.tensor().view(-1)
        
        t2[...] = torch.sum(t1, dim=1)
        
        return result
    
    def elem_mult(self, v1: torch.Tensor, v2: torch.Tensor, *, out: Path) -> TensorProxy:
        return self.element_wise_binary_operation(v1, v2, lambda t1, t2, t3: torch.mul(t1, t2, out=t3), out)
    
    def elem_div(self, v1: torch.Tensor, v2: torch.Tensor, *, out: Path) -> TensorProxy:
        return self.element_wise_binary_operation(v1, v2, lambda t1, t2, t3: torch.div(t1, t2, out=t3), out)
    
    def matrix_mult(self, v1: torch.Tensor, v2: torch.Tensor, *, out: Path) -> TensorProxy:
        if v1.dtype != v2.dtype:
            raise Exception()
        
        data_type = DataType.get_by_dtype(v1.dtype)
        manager = self.__managers[data_type.name]
        
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
        max_parallel_operations = manager.max_mem // mem_per_op
        if max_parallel_operations <= 0: raise Exception()
        
        t1 = v1.view(-1, rows, n)
        t2 = v2.view(-1, n, cols)
        result = TensorProxy.empty_override((*shape1[:-2], rows, cols), out, data_type)
        t3 = result.tensor().view(-1, rows, cols)
        
        g1, g2, g3 = manager.alloc_tensors( \
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
    
    def sqrt(self, v1: torch.Tensor, *, out: Path) -> TensorProxy:
        return self.element_wise_unary_operation(v1, lambda t1, t2: torch.sqrt(t1, out=t2), out)
    
    def get_norm(self, v1: torch.Tensor) -> float:
        return torch.linalg.vector_norm(v1)
    
    def divide_by_scalar(self, v1: torch.Tensor, divisor: float, *, out: Path) -> TensorProxy:
        return self.element_wise_unary_operation(v1, lambda t1, t2: torch.div(t1, divisor, out=t2), out)
    
    def normalize(self, v1: torch.Tensor, *, out: Path) -> TensorProxy:
        norm = self.get_norm(v1)
        return self.divide_by_scalar(v1, norm, out=out)
    
    def adjoint(self, t1: torch.Tensor, *, out: Path) -> TensorProxy:
        data_type = DataType.get_by_dtype(t1.dtype)
        shape = t1.shape
        rows = shape[-2]
        cols = shape[-1]
        
        result = TensorProxy.empty_override((*shape[:-2], cols, rows), out, data_type)
        t2 = result.tensor()
        t2[...] = torch.adjoint(t1)
        
        return result
    
    def real(self, t1: torch.Tensor, *, out: Path) -> TensorProxy:
        data_type = DataType.get_by_dtype(t1.dtype)
        result_type = data_type.to_real()
        
        result = TensorProxy.empty_override(v1.shape, out, result_type)
        t2 = result.tensor()
        
        t2[...] = t1.real
        
        return result
    
    def softmax(self, v1: torch.Tensor, *, out: Path) -> TensorProxy:
        data_type = DataType.get_by_dtype(v1.dtype)
        
        if not data_type.is_real():
            raise Exception()
        
        manager = self.__managers[data_type.name]
        
        shape = v1.shape
        
        vector_length = shape[-1]
        
        num_vectors = get_size(shape) // vector_length
        mem_per_op = vector_length * 2 + 1
        max_parallel_operations = manager.max_mem // mem_per_op
        if max_parallel_operations <= 0: raise Exception()
        
        t1 = v1.view(-1, vector_length)
        result = TensorProxy.empty_override(shape, out, data_type)
        t2 = result.tensor().view(-1, vector_length)
        
        if self.use_gpu:
            g1, g2, g_unit = manager.alloc_tensors( \
                (max_parallel_operations, *n) for n in ((vector_length,), (vector_length,), (1,)) \
            )
            
            start = 0
            while start < num_vectors:
                stop = min(num_vectors, start + max_parallel_operations)
                g01 = g1[:stop - start]
                g02 = g2[:stop - start]
                g0unit = g_unit[:stop - start]
                
                g01[...] = t1[start:stop]
                torch.amax(g01, dim=1, keepdim=True, out=g0unit)
                torch.sub(g01, g0unit, out=g02)
                torch.exp(g02, out=g02)
                torch.sum(g02, dim=1, keepdim=True, out=g0unit)
                torch.div(g02, g0unit, out=g02)
                
                t2[start:stop] = g02
                start = stop
        else:
            t_unit = torch.empty(num_vectors, 1, dtype=data_type.value)
            torch.amax(t1, dim=1, keepdim=True, out=t_unit)
            torch.sub(t1, t_unit, out=t2)
            torch.exp(t2, out=t2)
            torch.sum(t2, dim=1, keepdim=True, out=t_unit)
            torch.div(t2, t_unit, out=t2)
        
        return result
    
    def diagonal(self, t1: torch.Tensor, *, out: Path) -> TensorProxy:
        data_type = DataType.get_by_dtype(t1.dtype)
        shape = t1.shape
        if shape[-2] != shape[-1]:
            raise Exception()
        matrix_len = shape[-1]
        
        result = TensorProxy.empty_override(shape[:-1], out, data_type)
        t2 = result.tensor()
        for i in range(matrix_len):
            t2[..., i] = t1[..., i, i]
        
        return result

