from typing import TypeVar, TypeGuard, Callable

T = TypeVar('T')

def not_none(v: T | None) -> TypeGuard[T]:
    return v is not None

def value_or(v: T | None, default: T) -> T:
    return v if not_none(v) else default

def value_or_else(v: T | None, default_fn: Callable[[], T]) -> T:
    if not_none(v):
        return v
    
    return default_fn()
