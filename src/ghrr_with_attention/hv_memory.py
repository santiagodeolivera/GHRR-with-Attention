import numpy as np
from pathlib import Path

def get_random_hvs(depth: int, matrix_size: int, file_path: Path, length: int) -> np.ndarray:
    res: np.ndarray
    if not file_path.exists():
        rand = np.random.default_rng()
        res = rand.normal(loc=0, scale=1, size=(length, depth, matrix_size, matrix_size))
        np.save(file_path, res)
    else:
        res = np.load(file_path)
    
    return res
