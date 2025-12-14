import numpy as np
from ghrr_with_attention.utils import value_or

# HVs are represented as np.ndarray instances of complex numbers, in which the last three dimensions must be depth, row, and column, from first to last

def norm(data: np.ndarray) -> np.ndarray:
    dims = len(data.shape)
    norm = np.linalg.vector_norm(data, axis=(dims-3, dims-2, dims-1))
    return norm

def normalize(data: np.ndarray) -> np.ndarray:
    norm_ = norm(data)[:, np.newaxis, np.newaxis, np.newaxis]
    return np.divide(data, norm_, out=np.zeros(data.shape, dtype=np.complex128), where=norm != 0)

def add_grouped(data: np.ndarray, *, axis: tuple[int, ...] | None = None) -> np.ndarray:
    axis_: tuple[int, ...] = value_or(axis, tuple(range(len(data.shape) - 3)))

    for n in range(1, 4):
        axis_id = len(data.shape) - n
        if n in axis_:
            raise ValueError(F"Axis {axis_id} is internal to the structure of HVs")
    
    return np.sum(data, axis=axis_)

def mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a, b)

def bundle_grouped(data: np.ndarray, *, axis: tuple[int, ...] | None = None) -> np.ndarray:
    v = add_grouped(data, axis=axis)
    return normalize(v)

def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = mult(a, b)
    return normalize(v)

def query_from_encoded(positional_encodings: np.ndarray, encodings: np.ndarray) -> np.ndarray:
    v1 = mult(positional_encodings, encodings)
    v2 = add_grouped(v1)
    return normalize(v2)

def key_from_encoded(encodings1: np.ndarray, encodings2: np.ndarray, positions2: np.ndarray) -> np.ndarray:
    v1 = mult(encodings2, positions2)
    v2 = np.conjugate(np.transpose(v1, (0, 2, 1)))
    v3 = mult(v2, encodings1)
    return normalize(v3)

def value_from_encoded(positional_encodings: np.ndarray, encodings: np.ndarray) -> np.ndarray:
    v1 = mult(positional_encodings, encodings)
    v2 = add_grouped(v1)
    return normalize(v2)
