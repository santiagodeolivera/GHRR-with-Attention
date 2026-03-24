import time
from datetime import datetime

time_start = time.time_ns()
date_start = datetime.fromtimestamp(time_start // 1_000_000_000)

__all__ = ["time_start", "date_start"]
