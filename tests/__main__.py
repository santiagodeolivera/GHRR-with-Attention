import unittest
import numpy as np
from ghrr import hv_functions

# TODO: Make this work

def f1(n: int) -> int:
    v1 = sum(x**2 for x in range(n, n + 8))
    return np.sqrt(v1)

def f2(x: int, d: int, r: int, c: int) -> np.complex128:
    np.complex128(x * 8 + d * 4 + r * 2 + c)

class HvFunctionsTest(unittest.TestCase):
    def test_normalize(self):
        expected = tuple(f1(n) for n in range(5 * 8) if n % 8 == 0)

        array = np.fromfunction(f2, axis=(5, 2, 2, 2))
        result = hv_functions.normalize()

        self.assertEqual(result, expected)
