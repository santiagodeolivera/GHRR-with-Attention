from typing import Callable

import torch

from gpu_management.tensor_functions import TensorFunctionsManager
from gpu_management.data_type import DataType
from get_args import get_arg
from tudataset import get_dataset_info
from constants import D, m
import abc

max_num_nodes = get_dataset_info().max_num_nodes

class PosEncMode(abc.ABC):
    def check_available(self) -> None:
        pass
    
    @abc.abstractmethod
    def build_tensor(self, dims: tuple[torch.Tensor, ...], out: torch.Tensor) -> None: ...
    
    def get_position_encodings(self, manager: TensorFunctionsManager) -> torch.Tensor:
        mid_result = manager.new_from_function((max_num_nodes, m, m), DataType.complex64, self.build_tensor)
        position_encodings = mid_result[:, None, :, :].expand(max_num_nodes, D, m, m)
        
        return position_encodings

class PosEncMode1(PosEncMode):
    def check_available(self) -> None:
        if max_num_nodes > m:
            raise Exception(f"Cannot use position encoding mode 1 on a dataset whose max number of nodes in a graph is higher than {m}")
    
    def build_tensor(self, dims: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        const0 = torch.tensor(0.0, dtype=torch.float32)
        const1 = torch.tensor(1.0, dtype=torch.float32)
        
        n, row, col = dims
        
        result = torch.where((n == row) & (row == col), const1, const0)
        
        out[...] = result.type(torch.complex64)

class PosEncMode2(PosEncMode):
    def build_tensor(self, dims: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        const0 = torch.tensor(0.0, dtype=torch.float32)
        const1 = torch.tensor(1.0, dtype=torch.float32)
        
        n, row, col = dims
        
        result = torch.where(row == col, \
            torch.clamp(n * (m / max_num_nodes) - row, const0, const1), \
        const0)
        
        out[...] = result.type(torch.complex64)

pos_enc_modes: dict[int, PosEncMode] = {
    1: PosEncMode1(),
    2: PosEncMode2()
}

pos_enc_mode_id = get_arg("POS_ENC_MODE", "int")
try:
    pos_enc_mode = pos_enc_modes[pos_enc_mode_id]
except KeyError:
    raise Exception(f"Invalid positional encoding mode: {pos_enc_mode_id}")

pos_enc_mode.check_available()

position_encodings_cache: torch.Tensor | None = None
def get_position_encodings(manager: TensorFunctionsManager) -> torch.Tensor:
    global position_encodings_cache
    if position_encodings_cache is not None:
        return position_encodings_cache
    
    position_encodings = pos_enc_mode.get_position_encodings(manager)
    
    position_encodings_cache = position_encodings
    return position_encodings

__all__ = ["get_position_encodings"]

