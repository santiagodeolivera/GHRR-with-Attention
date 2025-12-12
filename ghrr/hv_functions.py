import numpy as np

# HVs are represented as np.ndarray instances of complex numbers, in which the last three dimensions must be depth, row and column, from first to last

def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize a group of HVs
    """

    dims = len(data.shape)
    norm = np.linalg.vector_norm(data, axis=(dims-3, dims-2, dims-1))
    norm = norm[:, np.newaxis, np.newaxis, np.newaxis]

    return np.divide(data, norm, out=np.zeros(data.shape, dtype=np.complex128), where=norm != 0)

def bundle(data: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    