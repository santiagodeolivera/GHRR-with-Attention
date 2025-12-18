from typing import Callable

class Experiment:
    fn: Callable[[], None]

    def __init__(self, fn: Callable[[], None]):
        self.fn = fn
    
    def execute(self):
        self.fn()
