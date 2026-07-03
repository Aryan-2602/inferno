"""
utils.py — Timing, memory measurement, and logging helpers for Inferno.

Provides reusable utilities for: wall-clock and GPU timing, peak memory tracking
(CPU and GPU), structured JSON result saving with timestamps, and a consistent
logging setup used across all benchmark and test scripts.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch

RESULTS_DIR = Path(__file__).parent.parent / "results"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the project-standard format."""
    return logging.getLogger(name)


def now_iso() -> str:
    """Return current UTC time as a compact ISO string safe for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_results(name: str, data: dict[str, Any]) -> Path:
    """
    Serialize data to results/<name>_<timestamp>.json and return the path.

    Never overwrites — every call appends a fresh timestamped file so benchmark
    history is preserved per CLAUDE.md rules.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}_{now_iso()}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def measure_memory_mb() -> float:
    """
    Return current memory usage in MB for the active device.

    On CUDA: reports peak allocated GPU memory since the last reset
    (torch.cuda.max_memory_allocated). Accurate to the byte for tensor
    allocations; does not include CUDA context overhead.

    On CPU: reports the process RSS (resident set size) via psutil.
    # TODO: GPU path gives allocated memory; CPU RSS includes full process
    # including model weights — document this difference. On CPU, the RSS
    # reflects all resident pages (model weights + KV cache + activations +
    # Python overhead), not just the incremental allocation from generate().
    # This is intentional: we want a number that reflects true system memory
    # pressure, not just the marginal cost of one call.
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return psutil.Process().memory_info().rss / (1024 ** 2)


class GpuMemoryTracker:
    """
    Context manager that records peak memory usage for the active device.

    On CUDA: resets peak stats on entry and reads max_memory_allocated on exit
    — gives the true peak tensor allocation during the block.

    On CPU: snapshots RSS on entry and exit and reports the higher of the two.
    RSS can only grow or stay flat within a normal inference call (no GC between
    steps), so the exit snapshot equals the peak for our use cases.

    Usage:
        with GpuMemoryTracker(device) as m:
            model.generate(...)
        print(m.peak_mb)  # hundreds of MB on CPU, model weights included
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.peak_mb: float = 0.0
        self._rss_before: float = 0.0

    def __enter__(self) -> "GpuMemoryTracker":
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            self._rss_before = psutil.Process().memory_info().rss / (1024 ** 2)
        return self

    def __exit__(self, *_: Any) -> None:
        if self.device.type == "cuda":
            self.peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        else:
            rss_after = psutil.Process().memory_info().rss / (1024 ** 2)
            # Take the higher of before/after — on CPU, RSS at end of generate()
            # includes model weights + full KV cache and is the meaningful number.
            self.peak_mb = max(rss_after, self._rss_before)


def wall_time() -> float:
    """Return a high-resolution wall-clock timestamp in seconds."""
    return time.perf_counter()
