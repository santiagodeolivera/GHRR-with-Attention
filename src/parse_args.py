import argparse
from dataclasses import dataclass
from typing import Literal

def new_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="GHRR with attention",
    )
    
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument("-p", "--program")
    parser.add_argument("-r", "--root-path")
    parser.add_argument("-D", "--dims")
    parser.add_argument("-m", "--matrix-size")
    
    return parser

@dataclass
class Args:
    program: str,
    root_path: str,
    dims: str,
    matrix_size: str

def get_arg(name: str, value: str | None) -> str:
    if value is None:
        raise Exception(f"Required parameter --{name}")
    
    return value

def get_args() -> Args | Literal["test"]:
    raw = new_parser().parse_args()
    
    if raw.test:
        return "test"
    
    raw_params: tuple[tuple[str, str | None], ...] = (
        ("program", raw.program),
        ("root-path", raw.root_path),
        ("dims", raw.dims),
        ("matrix-size", raw.matrix_size)
    )
    
    params = tuple(get_arg(name, value) for name, value in raw_params)
    return Args(program = params[0], root_path = params[1], dims = params[2], matrix_size = params[3])

args = get_args()

__all__ = ["args"]

