from dataclasses import dataclass
from fs_organization import FsOrganizer
from hv_functions import UpperTensorFunctionsManager

@dataclass
class FnContext:
    fs: FsOrganizer
    functions: UpperTensorFunctionsManager

