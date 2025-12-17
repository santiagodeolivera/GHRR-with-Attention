import unittest
import numpy as np
import torch
from ghrr_with_attention import hv_functions
from utils import torch_singular_value

def f1(n: int) -> int:
    v1 = sum(x**2 for x in range(n, n + 8))
    return np.sqrt(v1)

def f2() -> torch.Tensor:
    v1 = np.fromfunction(lambda x, d, r, c: 8*x + 4*d + 2*r + c, dtype=np.complex128, shape=(5, 2, 2, 2))
    return torch.from_numpy(v1)

class HvFunctionsTest(unittest.TestCase):
    def test_norm(self):
        expected = np.fromiter((f1(n) for n in range(5 * 8) if n % 8 == 0), dtype=np.float64)

        array = f2()
        result = hv_functions.norm(array)

        print(f"Expected: {expected}")
        print(f"Result: {result}")
        self.assertTrue((result == expected).all())
    
    def test_normalize(self):
        array = f2()
        result = hv_functions.normalize(array)
        norm = torch.linalg.vector_norm(result, dim=(1, 2, 3))

        print("---A---")
        print(array)
        print("---B---")
        print(result)
        
        proportion = array / result
        
        print("---C---")
        print(proportion)
        
        self.assertTrue(torch_singular_value(proportion, data_dims=3))
        self.assertTrue(torch.all(torch.isclose(1, norm)))

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

if __name__ == "__main__":
    unittest.main()
#
