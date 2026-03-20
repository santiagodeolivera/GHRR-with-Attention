from pathlib import Path
from typing import Callable, Any

import torch

from tensor_proxy import TensorProxy
from memory import MemoryManager
from utils import get_size
from constants import DataType

class TensorFunctionsManager:
    __manager: MemoryManager
    __real_n_manager: MemoryManager
    use_gpu: bool
    
    def __init__(self, manager: MemoryManager, real_n_manager: MemoryManager, use_gpu: bool = True):
        if not manager.data_type.is_complex:
            raise Exception()
            
        if real_n_manager.data_type.is_complex:
            raise Exception()
            
        self.__manager = manager
        self.__real_n_manager = real_n_manager
        self.use_gpu = use_gpu
    
    def randn(self, shape: tuple[int, ...], out: Path, data_type: DataType) -> TensorProxy:
        result = TensorProxy.empty(shape, out, data_type)
        tensor = result.tensor()
        tensor[...] = torch.randn(shape, dtype=data_type.value, device="cpu")
        return result
    
    def element_wise_unary_operation(self, \
        v1: TensorProxy, \
        fn: Callable[[torch.Tensor, torch.Tensor], Any], \
        out: Path
    ) -> TensorProxy:
        if v1.data_type != self.__manager.data_type:
            raise Exception()
        
        size = get_size(v1.shape)
        max_size = self.__manager.max_mem // 3
        if max_size <= 0: raise Exception()
        
        t1 = v1.tensor().view(-1)
        result = TensorProxy.empty(v1.shape, out, self.__manager.data_type)
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
        if v1.data_type != self.__manager.data_type or v2.data_type != self.__manager.data_type:
            raise Exception()
        
        if v1.shape != v2.shape:
            raise ValueError("Elements must have the same shape")
        
        size = get_size(v1.shape)
        max_size = self.__manager.max_mem // 3
        if max_size <= 0: raise Exception()
        
        t1 = v1.tensor().view(-1)
        t2 = v2.tensor().view(-1)
        result = TensorProxy.empty(v1.shape, out, self.__manager.data_type)
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
    
    def summation(self, v1: TensorProxy, unit_dims: int, out: Path) -> TensorProxy:
        if unit_dims < 0:
            raise Exception()
        
        shape = v1.shape
        if len(shape) < unit_dims:
            raise Exception()
        
        batch_shape = shape[:-unit_dims] if unit_dims != 0 else shape
        unit_shape  = shape[-unit_dims:] if unit_dims != 0 else ()
        
        batch_size = get_size(batch_shape)
        unit_size = get_size(unit_shape)
        
        t1 = v1.tensor().view(-1, unit_size)
        result = TensorProxy.empty(batch_shape, out, self.__manager.data_type)
        t2 = result.tensor().view(-1)
        
        t2[...] = torch.sum(t1, dim=1)
        
        return result
        
        """
        if unit_dims < 0:
            raise Exception()
        
        if v1.data_type != self.__manager.data_type:
            raise Exception()
        
        shape = v1.shape
        if len(shape) < unit_dims:
            raise Exception()
        
        batch_shape = shape[:-unit_dims] if unit_dims != 0 else shape
        unit_shape  = shape[-unit_dims:] if unit_dims != 0 else ()
        
        batch_size = get_size(batch_shape)
        unit_size = get_size(unit_shape)
        
        num_operations = batch_size
        mem_per_op = unit_size + 1
        max_parallel_operations = self.__manager.max_mem // mem_per_op
        if max_parallel_operations <= 0: raise Exception()
        
        t1 = v1.tensor().view(-1, unit_size)
        result = TensorProxy.empty(batch_size, out, self.__manager.data_type)
        t2 = result.tensor().view(-1)
        
        g1, g2 = self.__manager.alloc_tensors( \
            (max_parallel_operations, *n) for n in ((batch_size,), ()) \
        )
        
        start = 0
        while start < num_operations:
            stop = min(num_operations, start + max_parallel_operations)
            g01 = g1[:stop - start]
            g02 = g2[:stop - start]
            
            g01[...] = t1[start:stop]
            
            torch.sum(g01, out=g02)
            
            t2[start:stop] = g02
            start = stop
    
        return result
        """
    
    def matrix_mult(self, v1: TensorProxy, v2: TensorProxy, out: Path) -> TensorProxy:
        if v1.data_type != self.__manager.data_type or v2.data_type != self.__manager.data_type:
            raise Exception()
        
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
        result = TensorProxy.empty((*shape1[:-2], rows, cols), out, self.__manager.data_type)
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
        
        t1 = v1.tensor()
        result = TensorProxy.empty((*shape[:-2], cols, rows), out, self.__manager.data_type)
        t2 = result.tensor()
        t2[...] = torch.adjoint(t1)
        
        return result
    
    def real(self, v1: TensorProxy, out: Path) -> TensorProxy:
        t1 = v1.tensor()
        result = TensorProxy.empty(v1.shape, out, self.__real_n_manager.data_type)
        t2 = result.tensor()
        
        t2[...] = t1.real.type(self.__real_n_manager.data_type.value)
        
        return result
    
    def softmax(self, v1: TensorProxy, out: Path) -> TensorProxy:
        if v1.data_type != self.__real_n_manager.data_type:
            raise Exception()
        
        shape = v1.shape
        
        vector_length = shape[-1]
        
        num_vectors = get_size(shape) // vector_length
        mem_per_op = vector_length * 2 + 1
        max_parallel_operations = self.__real_n_manager.max_mem // mem_per_op
        if max_parallel_operations <= 0: raise Exception()
        
        t1 = v1.tensor().view(-1, vector_length)
        result = TensorProxy.empty(shape, out, self.__real_n_manager.data_type)
        t2 = result.tensor().view(-1, vector_length)
        
        if self.use_gpu:
            g1, g2, g_unit = self.__real_n_manager.alloc_tensors( \
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
            t_unit = torch.empty(num_vectors, 1, dtype=self.__real_n_manager.data_type.value)
            torch.amax(t1, dim=1, keepdim=True, out=t_unit)
            torch.sub(t1, t_unit, out=t2)
            torch.exp(t2, out=t2)
            torch.sum(t2, dim=1, keepdim=True, out=t_unit)
            torch.div(t2, t_unit, out=t2)
        
        return result
    
    def swap_dims(self, v1: TensorProxy, i1: int, i2: int, out: Path) -> TensorProxy:
        shape0 = list(v1.shape)
        shape0[i1], shape0[i2] = shape0[i2], shape0[i1]
        shape = tuple(shape0)
        
        t1 = v1.tensor()
        result = TensorProxy.empty(shape, out, self.__real_n_manager.data_type)
        t2 = result.tensor()
        
        t2[...] = t1.swapdims(i1, i2)
        
        return result

