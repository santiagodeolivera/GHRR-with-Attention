import torch
from pathlib import Path

def get_random_hvs(depth: int, matrix_size: int, file_path: Path, length: int) -> torch.Tensor:
    res: torch.Tensor
    if not file_path.exists():
        res = torch.randn(length, depth, matrix_size, matrix_size)
        
        print(f"Saving into {str(file_path)}")
        torch.save(res, file_path)
        print(f"Saved into {str(file_path)}")
    else:
        res = torch.load(file_path)
    
    return res
