import torch

default_device: torch.device
if torch.cuda.is_available():
    default_device = torch.device("cuda")
else:
    print("Warning: CUDA is not available. Using CPU")
    default_device = torch.device("cpu")

__all__ = ["default_device"]
