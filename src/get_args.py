import os
from typing import Callable, overload, Literal, Any
from pathlib import Path

type CastFn = Callable[[str], Any]

cast_fns: dict[str, CastFn] = {
    "str": lambda s: s,
    "int": lambda s: int(s),
    "float": lambda s: float(s),
    "Path": lambda s: Path(s)
}

@overload
def get_arg(name: str, cast: Literal["str"]) -> str: ...
@overload
def get_arg(name: str, cast: Literal["int"]) -> int: ...
@overload
def get_arg(name: str, cast: Literal["float"]) -> float: ...
@overload
def get_arg(name: str, cast: Literal["Path"]) -> Path: ...

def get_arg(name: str, cast: str) -> Any:
    value: str | None = os.environ.get(name, None)
    if value is None or len(value) == 0:
        raise Exception(f"Env var {repr(name)} not present")
    
    cast_fn: CastFn | None = cast_fns.get(cast, None)
    if cast_fn is None:
        raise Exception(f"Unexpected error: Invalid cast function: {repr(cast_fn)}")
    
    return cast_fn(value)

@overload
def get_op_arg(name: str, cast: Literal["str"]) -> str | None: ...
@overload
def get_op_arg(name: str, cast: Literal["int"]) -> int | None: ...
@overload
def get_op_arg(name: str, cast: Literal["float"]) -> float | None: ...
@overload
def get_op_arg(name: str, cast: Literal["Path"]) -> Path | None: ...

def get_op_arg(name: str, cast: str) -> Any:
    value: str | None = os.environ.get(name, None)
    if value is None or len(value) == 0:
        return None
    
    cast_fn: CastFn | None = cast_fns.get(cast, None)
    if cast_fn is None:
        raise Exception(f"Unexpected error: Invalid cast function: {repr(cast_fn)}")
    
    return cast_fn(value)

__all__ = ["get_arg", "get_op_arg"]

