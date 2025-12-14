import numpy as np
from pathlib import Path

class SeparatedRandomMemory:
    """
    A Random HV memory that stores HVs in separate files in a specific directory of .npy files
    """

    depth: int
    matrix_size: int
    storage_path: Path
    rand: np.random.Generator

    def __init__(self, depth: int, matrix_size: int, storage_path: Path):
        if not storage_path.is_dir():
            raise ValueError("A directory path is necessary for storage")
        
        self.storage_path = storage_path
        self.rand = np.random.default_rng()
    
    def get(self, key: int) -> np.ndarray:
        file_path = self.storage_path / f"{key}.npy"

        res: np.ndarray
        if not file_path.exists():
            res = self.rand.normal(loc=0, scale=1, size=(self.depth, self.matrix_size, self.matrix_size))
            np.save(file_path, res)
        else:
            res = np.load(file_path)
        
        return res

class JoinedRandomMemory:
    """
    A Random HV memory that prepares all HVs in advance, and stores them in a .npy file
    """

    array: np.ndarray

    def __init__(self, depth: int, matrix_size: int, file_path: Path, limit_hvs: int):
        if not file_path.is_file():
            raise ValueError("A file path is necessary for storage")
        
        array: np.ndarray
        if not file_path.exists():
            rand = np.random.default_rng()
            array = self.rand.normal(loc=0, scale=1, size=(limit_hvs, self.depth, self.matrix_size, self.matrix_size))
            np.save(file_path, array)
        else:
            array = np.load(file_path)
        
        self.array = array
    
    def get(self, key: int) -> np.ndarray:
        return self.array[key, :, :, :]
