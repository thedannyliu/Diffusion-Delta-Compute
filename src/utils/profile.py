from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from typing import Dict, Optional
import threading


def query_gpu_utilization() -> Optional[Dict[str, float]]:
    """Best-effort GPU utilization query.

    Tries NVML via pynvml first, then falls back to `nvidia-smi` query. Returns None if both fail.
    """
    # Try NVML
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        out = {
            "util_gpu": float(util.gpu),
            "util_mem": float(util.memory),
            "mem_used": float(mem.used) / (1024.0 * 1024.0),
            "mem_total": float(mem.total) / (1024.0 * 1024.0),
        }
        pynvml.nvmlShutdown()
        return out
    except Exception:
        pass

    # Fallback to nvidia-smi --query
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ], stderr=subprocess.DEVNULL)
        line = out.decode("utf-8").strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        util_gpu = float(parts[0])
        util_mem = float(parts[1])
        mem_used = float(parts[2])
        mem_total = float(parts[3])
        return {"util_gpu": util_gpu, "util_mem": util_mem, "mem_used": mem_used, "mem_total": mem_total}
    except Exception:
        return None


class GPUUtilSampler:
    """GPU utilization sampler.

    - Background mode: periodic sampling in a thread.
    - Manual mode: call `sample_once()` at interesting points (e.g., after GPU ops).

    Aggregates running mean for: util_gpu, util_mem, mem_used, mem_total.
    """

    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval_sec = float(interval_sec)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._sum: Dict[str, float] = {}
        self._count: int = 0
        self.metrics: Dict[str, float] = {}

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            time.sleep(self.interval_sec)

    def sample_once(self) -> None:
        """Take a single instantaneous sample and fold into aggregates."""
        data = query_gpu_utilization()
        if data is None:
            return
        self._count += 1
        for key, value in data.items():
            self._sum[key] = self._sum.get(key, 0.0) + float(value)
        if self._count > 0:
            self.metrics = {k: self._sum[k] / float(self._count) for k in self._sum}

    def start(self) -> None:
        if self._thread is None:
            self._stop.clear()
            self._sum.clear()
            self._count = 0
            self.metrics = {}
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None


@contextmanager
def latency_timer() -> Dict[str, float]:
    start = time.time()
    info: Dict[str, float] = {}
    try:
        yield info
    finally:
        info["ms"] = (time.time() - start) * 1000.0
