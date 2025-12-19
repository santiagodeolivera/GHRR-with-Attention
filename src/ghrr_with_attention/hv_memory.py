import torch
from pathlib import Path
from ghrr_with_attention.utils import checkpoint_print

def get_random_hvs(depth: int, matrix_size: int, file_path: Path, length: int, *, device: torch.device) -> torch.Tensor:
    res: torch.Tensor
    if not file_path.exists():
        res = torch.randn(length, depth, matrix_size, matrix_size, device=device)
        
        checkpoint_print(f"Saving into {str(file_path)}")
        torch.save(res, file_path)
        checkpoint_print(f"Saved into {str(file_path)}")
    else:
        checkpoint_print(f"Retrieving from {str(file_path)}")
        res = torch.load(file_path, map_location=device)
        checkpoint_print(f"Retrieved from {str(file_path)}")
    
    return res
