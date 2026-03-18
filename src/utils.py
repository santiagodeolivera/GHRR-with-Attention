from typing import Callable, TypeVar, Any, TypeGuard
from functools import reduce

T = TypeVar("T")

def block(fn: Callable[[], T]) -> T:
    return fn()

def get_size(shape: tuple[int, ...]) -> int:
    return reduce(lambda a, b: a * b, shape, 1)

__all__ = ["block", "get_size"]

