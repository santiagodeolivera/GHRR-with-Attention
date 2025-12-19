from typing import Callable, Iterable

class Experiment:
    fn: Callable[[Iterable[str]], None]

    def __init__(self, fn: Callable[[Iterable[str]], None]):
        self.fn = fn
    
    def execute(self, args: Iterable[str]):
        self.fn(args)
