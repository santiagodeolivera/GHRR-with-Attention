import torch

if torch.cuda.is_available():
    device: torch.device = torch.device("cuda")
else:
    print("Warning: CUDA is not available. Using CPU")
    device: torch.device = torch.device("cpu")

__all__ = ["device"]
