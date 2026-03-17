from typing import Callable, TypeVar

T = TypeVar("T")

def block(fn: Callable[[], T]) -> T:
    return fn()

