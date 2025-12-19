import torch
from pathlib import Path
from ghrr_with_attention.utils import CheckpointContext

def get_random_hvs(depth: int, matrix_size: int, file_path: Path, length: int, *, device: torch.device, override: bool = False) -> torch.Tensor:
    res: torch.Tensor
    
    ctx: CheckpointContext = CheckpointContext(f"Random HVs from file {file_path}")
    
    if not file_path.exists() or override:
        res = torch.randn(length, depth, matrix_size, matrix_size, device=device)
        
        ctx.print("Saving (override)" if override else "Saving")
        torch.save(res, file_path)
        ctx.print("Saved (override)" if override else "Saved")
    else:
        ctx.print("Retrieving")
        res = torch.load(file_path, map_location=device)
        ctx.print("Retrieved")
    
    return res
