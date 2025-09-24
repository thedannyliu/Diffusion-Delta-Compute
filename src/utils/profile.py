from __future__ import annotations

import os
import subprocess
import time
from contextlib import contextmanager
from typing import Dict, Optional


def query_gpu_utilization() -> Optional[Dict[str, float]]:
    try:
        # Use nvidia-smi to fetch utilization for GPU 0
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


@contextmanager
def latency_timer() -> Dict[str, float]:
    start = time.time()
    info: Dict[str, float] = {}
    try:
        yield info
    finally:
        info["ms"] = (time.time() - start) * 1000.0


