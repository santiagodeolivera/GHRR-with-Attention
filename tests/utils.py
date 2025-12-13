import numpy as np
import unittest

def np_singular_value(data: np.ndarray, *, axis: tuple[int, ...]) -> bool:
    element_wise = np.isclose(np.max(data, axis=axis), np.min(data, axis=axis), equal_nan=True)
    return np.all(element_wise)
