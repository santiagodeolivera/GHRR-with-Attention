import os
from typing import Callable, overload, Literal, Any
from pathlib import Path

type CastFn = Callable[[str, str], Any]

def parse_to_bool(s: str, name: str) -> bool:
    if s == "0": return False
    if s == "1": return True
    raise Exception(f"Expected 0 or 1 in env var {repr(name)}")

cast_fns: dict[str, CastFn] = {
    "str": lambda s, n: s,
    "int": lambda s, n: int(s),
    "float": lambda s, n: float(s),
    "Path": lambda s, n: Path(s),
    "bool": parse_to_bool
}

@overload
def get_arg(name: str, cast: Literal["str"]) -> str: ...
@overload
def get_arg(name: str, cast: Literal["int"]) -> int: ...
@overload
def get_arg(name: str, cast: Literal["float"]) -> float: ...
@overload
def get_arg(name: str, cast: Literal["Path"]) -> Path: ...
@overload
def get_arg(name: str, cast: Literal["bool"]) -> bool: ...

def get_arg(name: str, cast: str) -> Any:
    value: str | None = os.environ.get(name, None)
    if value is None or len(value) == 0:
        raise Exception(f"Env var {repr(name)} not present")
    
    cast_fn: CastFn | None = cast_fns.get(cast, None)
    if cast_fn is None:
        raise Exception(f"Unexpected error: Invalid cast function: {repr(cast_fn)}")
    
    return cast_fn(value, name)

@overload
def get_op_arg(name: str, cast: Literal["str"]) -> str | None: ...
@overload
def get_op_arg(name: str, cast: Literal["int"]) -> int | None: ...
@overload
def get_op_arg(name: str, cast: Literal["float"]) -> float | None: ...
@overload
def get_op_arg(name: str, cast: Literal["Path"]) -> Path | None: ...
@overload
def get_op_arg(name: str, cast: Literal["bool"]) -> bool | None: ...

def get_op_arg(name: str, cast: str) -> Any:
    value: str | None = os.environ.get(name, None)
    if value is None or len(value) == 0:
        return None
    
    cast_fn: CastFn | None = cast_fns.get(cast, None)
    if cast_fn is None:
        raise Exception(f"Invalid cast function: {repr(cast)}")
    
    return cast_fn(value, name)

__all__ = ["get_arg", "get_op_arg"]

