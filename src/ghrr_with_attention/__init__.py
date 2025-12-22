import torch

device: torch.device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("Warning: CUDA is not available. Using CPU")
    device = torch.device("cpu")

__all__ = ["device"]
