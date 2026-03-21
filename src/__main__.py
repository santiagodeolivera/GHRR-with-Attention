from parse_args import args

from typing import Callable, Literal
from pathlib import Path

import torch

from programs.tests import all_tests
from programs.create_record import func as create_record

type Program = Callable[[Path], None]

programs: dict[str, Program] = {
    "1": create_record
}

def main():
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available")
    
    if args == "test":
        all_tests()
        return
    
    program_id = args.program
    program_data = programs.get(program_id, None)
    if program_data is None:
        raise Exception(f"Unknown program id: {repr(program_id)}")
    
    if program_data[0]:
        root_path_str = args.root_path
        
        root_path = Path(root_path_str)
        program_data[1](root_path)
    else:
        program_data[0]()

if __name__ == "__main__":
    main()

