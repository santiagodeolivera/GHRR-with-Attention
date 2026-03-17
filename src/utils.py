from typing import Callable, TypeVar, Any, TypeGuard

T = TypeVar("T")

def block(fn: Callable[[], T]) -> T:
    return fn()

