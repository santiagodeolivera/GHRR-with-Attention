from pathlib import Path
from typing import Callable, Any
from builtins import slice

import torch

from .tensor_proxy import TensorProxy
from .memory import MemoryManager
from .data_type import DataType
from utils import get_size, get_range_tensor

class TensorFunctionsManager:
    __managers: dict[str, MemoryManager]
    
    def __init__(self, max_bytes: int):
        max_mem = max_bytes // DataType.complex64.size
        
        complex_manager, float_manager = MemoryManager.create_two(max_mem)
        
        self.__managers = {
            "complex64": complex_manager,
            "float32": float_manager
        }
    
    def __enter__(self) -> "TensorFunctionsManager":
        return self
    
    def __exit__(self, exc_t, exc_v, exc_tb) -> None:
        for manager in self.__managers.values():
            manager.__exit__(exc_t, exc_v, exc_tb)
    
    def randn(self, shape: tuple[int, ...], data_type: DataType, *, out: Path) -> TensorProxy:
        manager = self.__managers[data_type.name]
        
        size = get_size(shape)
        max_size = manager.max_mem
        if max_size <= 0: raise Exception()
        
        result = TensorProxy.empty_override(shape, out, data_type)
        tensor = result.tensor().reshape(-1)
        
        (g1,) = manager.alloc_tensors(((max_size,),))
        
        start = 0
        while start < size:
            stop = min(size, start + max_size)
            g01 = g1[:stop - start]
            
            torch.randn(stop - start, out=g01)
            
            tensor[start:stop] = g01
            start = stop
        
        return result
    
    def new_from_function(self, shape: tuple[int, ...], data_type: DataType, \
        function: Callable[[tuple[torch.Tensor, ...]], torch.Tensor], *, out: Path) -> TensorProxy:
        result = TensorProxy.empty_override(shape, out, data_type)
        tensor = result.tensor()
        
        dimensions_range = tuple(get_range_tensor(n) for n in shape)
        dimensions_mesh = tuple(torch.meshgrid(*dimensions_range, indexing="ij"))
        
        tensor[...] = function(dimensions_mesh)
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
        
        t1 = v1.reshape(-1)
        result = TensorProxy.empty_override(v1.shape, out, data_type)
        t2 = result.tensor().reshape(-1)
        
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
        
        t1 = v1.reshape(-1)
        t2 = v2.reshape(-1)
        result = TensorProxy.empty_override(v1.shape, out, data_type)
        t3 = result.tensor().reshape(-1)
        
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
        
        return result

    def element_wise_binary_operation_self_modify(self, \
        v1: torch.Tensor, \
        v2: torch.Tensor, \
        fn: Callable[[torch.Tensor, torch.Tensor], Any] \
    ) -> torch.Tensor:
        if v1.dtype != v2.dtype:
            raise Exception()
        
        data_type = DataType.get_by_dtype(v1.dtype)
        manager = self.__managers[data_type.name]
        
        if v1.shape != v2.shape:
            raise ValueError("Elements must have the same shape")
        
        size = get_size(v1.shape)
        max_size = manager.max_mem // 2
        if max_size <= 0: raise Exception()
        
        t1 = v1.reshape(-1)
        t2 = v2.reshape(-1)
        
        g1, g2 = manager.alloc_tensors((max_size,) for _ in range(2))
        
        start = 0
        while start < size:
            stop = min(size, start + max_size)
            g01 = g1[:stop - start]
            g02 = g2[:stop - start]
            
            g01[...] = t1[start:stop]
            g02[...] = t2[start:stop]
            
            fn(g01, g02)
            
            t1[start:stop] = g01
            start = stop
        
        return t1
    
    def addition(self, v1: torch.Tensor, v2: torch.Tensor, *, out: Path) -> TensorProxy:
        return self.element_wise_binary_operation(v1, v2, lambda t1, t2, t3: torch.add(t1, t2, out=t3), out)
    
    def assign_addition(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        return self.element_wise_binary_operation_self_modify(v1, v2, lambda t1, t2: torch.add(t1, t2, out=t1))
    
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
        
        t1 = v1.reshape(-1, unit_size)
        result = TensorProxy.empty_override(batch_shape, out, data_type)
        t2 = result.tensor().reshape(-1)
        
        t2[...] = torch.sum(t1, dim=1)
        
        return result
    
    def elem_mult(self, v1: torch.Tensor, v2: torch.Tensor, *, out: Path) -> TensorProxy:
        return self.element_wise_binary_operation(v1, v2, lambda t1, t2, t3: torch.mul(t1, t2, out=t3), out)
    
    def elem_div(self, v1: torch.Tensor, v2: torch.Tensor, *, out: Path) -> TensorProxy:
        return self.element_wise_binary_operation(v1, v2, lambda t1, t2, t3: torch.div(t1, t2, out=t3), out)
    
    def matrix_mult(self, v1: torch.Tensor, v2: torch.Tensor, *, out: Path) -> TensorProxy:
        if v1.dtype != v2.dtype:
            raise Exception(f"The dtypes of the two tensors do not match ({v1.dtype} and {v2.dtype})")
        
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
        
        t1 = v1.reshape(-1, rows, n)
        t2 = v2.reshape(-1, n, cols)
        result = TensorProxy.empty_override((*shape1[:-2], rows, cols), out, data_type)
        t3 = result.tensor().reshape(-1, rows, cols)
        
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
        
        result = TensorProxy.empty_override(t1.shape, out, result_type)
        t2 = result.tensor()
        
        t2[...] = t1.real
        
        return result
    
    def softmax(self, v1: torch.Tensor, *, out: Path) -> TensorProxy:
        if v1.dtype != torch.float32:
            raise Exception("Softmax function invalid for any type other than float32")
        
        manager = self.__managers["float32"]
        
        shape = v1.shape
        
        vector_length = shape[-1]
        
        num_vectors = get_size(shape) // vector_length
        mem_per_op = vector_length * 2 + 1
        max_parallel_operations = manager.max_mem // mem_per_op
        if max_parallel_operations <= 0: raise Exception()
        
        t1 = v1.reshape(-1, vector_length)
        result = TensorProxy.empty_override(shape, out, DataType.float32)
        t2 = result.tensor().reshape(-1, vector_length)
        
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

