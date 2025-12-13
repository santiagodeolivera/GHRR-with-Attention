import unittest
import numpy as np
from ghrr_with_attention import hv_functions
from utils import np_singular_value

def f1(n: int) -> int:
    v1 = sum(x**2 for x in range(n, n + 8))
    return np.sqrt(v1)

def f2(x: int, d: int, r: int, c: int) -> np.ndarray:
    return (x * 8 + d * 4 + r * 2 + c).astype(np.complex128)

class HvFunctionsTest(unittest.TestCase):
    def test_norm(self):
        expected = np.fromiter((f1(n) for n in range(5 * 8) if n % 8 == 0), dtype=np.float64)

        array = np.fromfunction(f2, shape=(5, 2, 2, 2))
        result = hv_functions.norm(array)

        print(f"Expected: {expected}")
        print(f"Result: {result}")
        self.assertTrue((result == expected).all())
    
    def test_normalize(self):
        array = np.fromfunction(f2, shape=(5, 2, 2, 2))
        result = hv_functions.normalize(array)
        norm = np.linalg.vector_norm(result, axis=(1, 2, 3))

        proportion = array / result
        self.assertTrue(np_singular_value(proportion, axis=(1, 2, 3)))
        self.assertTrue(np.all(np.isclose(1, norm)))

class UtilsTest(unittest.TestCase):
    def test_singular_value_1(self):
        array = np.fromfunction(lambda a, b, c, d: a, shape=(5, 2, 2, 2))
        self.assertTrue(np_singular_value(array, axis=(1, 2, 3)))

    def test_singular_value_2(self):
        array = np.fromfunction(lambda a, b, c, d: a + b, shape=(5, 2, 2, 2))
        self.assertFalse(np_singular_value(array, axis=(1, 2, 3)))

if __name__ == "__main__":
    unittest.main()
#
