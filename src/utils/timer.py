import time
from typing import Optional


class Timer:
    def __init__(self, name: str = "timer") -> None:
        self.name = name
        self.start_time: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            print(f"[{self.name}] Elapsed: {elapsed:.3f}s")


