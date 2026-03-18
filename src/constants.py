from builtins import slice

import torch

element_type = torch.complex64
element_size = 8

type SliceInfo = tuple[slice[int, int, int], ...]

