import time
time_start = time.time_ns()

from datetime import datetime

def calc_time_difference(before: int, after: int) -> str:
    time_difference = (after - before) // 10_000_000

    if time_difference < 0:
        time_difference *= -1
    
    time_difference_int = time_difference // 100
    time_difference_dec = time_difference  % 100

    return f"{time_difference_int}.{time_difference_dec:02} s"

def get_hour_and_minute(timestamp: int) -> str:
    seconds = timestamp // 1_000_000_000
    date = datetime.fromtimestamp(seconds)
    return f"{date.hour:02}:{date.minute:02}"

class Timer:
    start: int
    name: str
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.start = time.time_ns()
        
        self.__print(self.start, f"{self.name} -> START")
    
    def __print(self, time: int, msg: str) -> None:
        v1 = get_hour_and_minute(time)
        v2 = calc_time_difference(time_start, time)
        v3 = get_hour_and_minute(time_start)
        print(f"({v1}, {v2} since program start at {v3}) {msg}")
    
    def end(self) -> None:
        end = time.time_ns()
        self.__print(end, f"{self.name} -> END (took {calc_time_difference(self.start, end)})")
    
    def error(self) -> None:
        end = time.time_ns()
        self.__print(end, f"{self.name} -> ERROR (took {calc_time_difference(self.start, end)})")

__all__ = ["time_start", "date_start", "get_hour_and_minute", "Timer"]

