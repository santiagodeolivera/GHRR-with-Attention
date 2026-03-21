import argparse
from typing import Callable, Literal
from pathlib import Path

import torch

from programs.tests import all_tests
from programs.create_record import func as create_record

def new_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="GHRR with attention",
    )
    
    parser.add_argument("-p", "--program", required=True)
    parser.add_argument("-r", "--root-path")
    
    return parser

type Program = tuple[Literal[False], Callable[[], None]] | tuple[Literal[True], Callable[[Path], None]]

programs: dict[str, Program] = {
    "test": (False, all_tests),
    "create": (True, create_record)
}

def main():
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available")
    
    args = new_parser().parse_args()
    
    program_id = args.program
    program_data = programs.get(program_id, None)
    if program_data is None:
        raise Exception(f"Unknown program id: {repr(program_id)}")
    
    if program_data[0]:
        root_path_str = args.root_path
        if root_path_str is None:
            raise Exception(f"Root path required")
        
        root_path = Path(root_path_str)
        program_data[1](root_path)
    else:
        program_data[0]()

if __name__ == "__main__":
    main()

