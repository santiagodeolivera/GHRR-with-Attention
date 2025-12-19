import time
from typing import TypeVar, TypeGuard, Callable

# TODO: Find out where these functions should be located

T = TypeVar('T')

def not_none(v: T | None) -> TypeGuard[T]:
    return v is not None

def value_or(v: T | None, default: T) -> T:
    return v if not_none(v) else default

def value_or_else(v: T | None, default_fn: Callable[[], T]) -> T:
    if not_none(v):
        return v
    
    return default_fn()

def calc_time_difference(before: int, after: int):
    time_difference = (after - before) // 10_000_000

    if time_difference < 0:
        time_difference *= -1
    
    time_difference_int = time_difference // 100
    time_difference_dec = time_difference  % 100

    return f"{time_difference_int}.{time_difference_dec:02} s"

"""
start_time = time.time_ns()
last_time = start_time
def checkpoint_print(*args, **kwargs):
    global start_time
    global last_time
    
    current_time = time.time_ns()
    diff_from_start = calc_time_difference(start_time, current_time)
    diff_from_last = calc_time_difference(last_time, current_time)
    last_time = current_time
    print()
    print(diff_from_last, "since last checkpoint")
    print(diff_from_start, "since program start")
    print(*args, **kwargs)

def checkpoint_log(msg: str, value: T) -> T:
    checkpoint_print(msg)
    return value
"""

def commutative_cantor_pairing(a: int, b: int) -> int:
    if a > b: (a, b) = (b, a)
    a = a * 2 if a >= 0 else a * -2 - 1
    b = b * 2 if b >= 0 else b * -2 - 1
    return (a + b) * (a + b + 1) // 2 + b

class CheckpointContext:
    name: str
    start_time: int
    last_time: int
    
    def __init__(self, name: str, *, msg: str | None = None):
        self.start_time = time.time_ns()
        self.last_time = self.start_time
        
        self.name = name
        
        if msg is not None:
            self.print(msg)
        
    def print(self, msg: str):
        current_time = time.time_ns()
        diff_from_start = calc_time_difference(self.start_time, current_time)
        diff_from_last = calc_time_difference(self.last_time, current_time)
        self.last_time = current_time
        print()
        print(f"Checkpoint context: {self.name}")
        print(diff_from_last, "since last checkpoint")
        print(diff_from_start, f"since checkpoint context definition")
        print(msg)

    def log(self, msg: str, value: T) -> T:
        self.print(msg)
        return value
