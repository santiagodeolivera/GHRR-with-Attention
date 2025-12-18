import torch
from pathlib import Path

def get_random_hvs(depth: int, matrix_size: int, file_path: Path, length: int) -> torch.Tensor:
    res: torch.Tensor
    if not file_path.exists():
        rand = np.random.default_rng()
        res = rand.normal(loc=0, scale=1, size=(length, depth, matrix_size, matrix_size))
        torch.save(file_path, res)
    else:
        res = torch.load(file_path)
    
    return res
