import unittest
import numpy as np
import torch
from ghrr_with_attention import hv_functions, device as tensor_device
from ghrr_with_attention.utils import torch_cantor_pairing, cantor_pairing
from utils import torch_singular_value

def f1(n: int) -> int:
    v1 = sum(x**2 for x in range(n, n + 8))
    return np.sqrt(v1)

def f2() -> torch.Tensor:
    t_5 = torch.tensor(range(5), dtype=torch.int32, device=tensor_device)
    t_2 = torch.tensor(range(2), dtype=torch.int32, device=tensor_device)
    x, d, r, c = torch.meshgrid(t_5, t_2, t_2, t_2, indexing="ij")
    return (8*x + 4*d + 2*r + c).to(torch.complex128)

def f3() -> torch.Tensor:
    return torch.tensor(range(5), dtype=torch.complex128, device=tensor_device).expand(-1, 2, 2, 2)

class UtilsTest(unittest.TestCase):
    def test_singular_value_1(self):
        array = np.fromfunction(lambda a, b, c, d: a, shape=(5, 2, 2, 2))
        tensor = torch.from_numpy(array)
        self.assertTrue(torch_singular_value(tensor, data_dims=3))

    def test_singular_value_2(self):
        array = np.fromfunction(lambda a, b, c, d: a + b, shape=(5, 2, 2, 2))
        tensor = torch.from_numpy(array)
        self.assertFalse(torch_singular_value(tensor, data_dims=3))

    def test_singular_value_2(self):
        array = np.fromfunction(lambda a, b, c, d: a + b, shape=(5, 2, 2, 2))
        tensor = torch.from_numpy(array)
        self.assertTrue(torch_singular_value(tensor, data_dims=2))
    
    def test_torch_gather(self):
        t1 = torch.tensor(range(3), dtype=torch.int32)
        t2, t3, t4 = torch.meshgrid(t1, t1, t1, indexing="ij")
        t5 = torch_cantor_pairing(torch_cantor_pairing(t2, t3), t4)
        
        for d in range(3):
            for row in range(3):
                for col in range(3):
                    self.assertEqual(cantor_pairing(cantor_pairing(d, row), col), t5[d, row, col])
        
        t6 = torch.tensor(((0, 1, 2), (1, 2, 0), (2, 0, 1)))
        t6_view = t6.view(9)
        
        t7_mid = torch.gather(t5, dim=0, index=t6_view[..., None, None].expand(-1, 3, 3))
        t7 = t7_mid.view(3, 3, 3, 3)
        for a in range(3):
            for b in range(3):
                for row in range(3):
                    for col in range(3):
                        index = t6[a, b]
                        expected = t5[index, row, col]
                        result = t7[a, b, row, col]
                        self.assertEqual(expected, result)
    
    def test_torch_gather_2(self):
        n_vectors: int = 100
        vector_shape: tuple[int, ...] = (100, 100, 100)
        vector_min: int = -25
        vector_max: int = 25
        n_indices: int = 80
        
        t1: torch.Tensor = (torch.rand(n_vectors, *vector_shape, device=tensor_device) * (vector_max - vector_min) + vector_min).to(torch.int32)
        t2: torch.Tensor = (torch.rand(n_indices, device=tensor_device) * n_vectors).to(torch.int32)
        
        t3: torch.Tensor = torch.gather(t1, dim=0, index=t2[..., *((None,) * len(vector_shape))].expand(-1, *vector_shape))

        for i in range(n_indices):
            self.assertTrue((t3[i] == t1[t2[i]]).all())

if __name__ == "__main__":
    unittest.main()
#
