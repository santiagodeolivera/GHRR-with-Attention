import torch
from pathlib import Path

from utils import CheckpointContext
from hv_functions import normalize
import localTypes

def get_complex_random_hvs(depth: int, matrix_size: int, file_path: Path, length: int, *, device: torch.device, override: bool = False, out: torch.Tensor | None = None) -> torch.Tensor:
    res: torch.Tensor
    
    ctx: CheckpointContext = CheckpointContext(f"Random HVs from file {file_path}")
    
    fileExist: bool = file_path.exists()
    if not fileExist or override:
        override = fileExist
        
        real = torch.randn(length, depth, matrix_size, matrix_size, device=device, dtype=localTypes.encodeRealType)
        img = torch.randn(length, depth, matrix_size, matrix_size, device=device, dtype=localTypes.encodeRealType)
        complex = torch.complex(real, img, out=out)
        res = normalize(complex)
        
        ctx.print("Saving (override)" if override else "Saving")
        torch.save(res, file_path)
        ctx.print("Saved (override)" if override else "Saved")
    else:
        ctx.print("Retrieving")
        mid_res = torch.load(file_path, map_location="cpu")
        if out is None:
            res = mid_res.to(device)
        else:
            out[...] = mid_res
            res = out
        ctx.print("Retrieved")
    
    return res

