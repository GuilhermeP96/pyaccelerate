"""
pyaccelerate.memory — Memory monitoring, pressure detection & pool allocator.

Provides:
  - Real-time memory pressure detection (LOW / MEDIUM / HIGH / CRITICAL)
  - Automatic worker-count clamping under memory pressure
  - Simple slab/pool allocator for reusable byte buffers
  - RSS / VMS tracking for the current process
"""

from __future__ import annotations

import logging
import os
import threading
from enum import Enum, auto
from typing import List, Optional, Tuple

log = logging.getLogger("pyaccelerate.memory")


# ═══════════════════════════════════════════════════════════════════════════
#  Memory pressure levels
# ═══════════════════════════════════════════════════════════════════════════

class Pressure(Enum):
    """System memory pressure level."""
    LOW = auto()        # > 50% free
    MEDIUM = auto()     # 25-50% free
    HIGH = auto()       # 10-25% free
    CRITICAL = auto()   # < 10% free


def get_pressure() -> Pressure:
    """Determine current system memory pressure."""
    try:
        import psutil  # type: ignore[import-untyped]
        vm = psutil.virtual_memory()
        avail_pct = vm.available / vm.total * 100
    except ImportError:
        return Pressure.LOW  # can't tell → assume fine

    if avail_pct > 50:
        return Pressure.LOW
    if avail_pct > 25:
        return Pressure.MEDIUM
    if avail_pct > 10:
        return Pressure.HIGH
    return Pressure.CRITICAL


def get_stats() -> dict[str, float]:
    """Return memory stats in GB for the system and current process."""
    stats: dict[str, float] = {}
    try:
        import psutil  # type: ignore[import-untyped]
        vm = psutil.virtual_memory()
        stats["system_total_gb"] = vm.total / (1024 ** 3)
        stats["system_available_gb"] = vm.available / (1024 ** 3)
        stats["system_used_pct"] = vm.percent

        proc = psutil.Process()
        mem = proc.memory_info()
        stats["process_rss_gb"] = mem.rss / (1024 ** 3)
        stats["process_vms_gb"] = mem.vms / (1024 ** 3)
    except ImportError:
        stats["error"] = -1.0
    return stats


def clamp_workers(desired: int, floor: int = 1) -> int:
    """Reduce *desired* worker count if memory pressure is high.

    Returns the clamped value (never below *floor*).
    """
    pressure = get_pressure()
    if pressure == Pressure.LOW:
        return max(floor, desired)
    if pressure == Pressure.MEDIUM:
        return max(floor, desired * 3 // 4)
    if pressure == Pressure.HIGH:
        return max(floor, desired // 2)
    # CRITICAL
    return floor


# ═══════════════════════════════════════════════════════════════════════════
#  Buffer pool (slab allocator)
# ═══════════════════════════════════════════════════════════════════════════

class BufferPool:
    """Thread-safe pool of reusable ``bytearray`` buffers.

    Useful for I/O-heavy workloads that repeatedly allocate and release
    fixed-size buffers (e.g., file hashing, network transfers).

    Usage::

        pool = BufferPool(buffer_size=1_048_576, max_buffers=16)
        buf = pool.acquire()
        try:
            # ... use buf ...
            pass
        finally:
            pool.release(buf)
    """

    def __init__(self, buffer_size: int = 1 << 20, max_buffers: int = 16):
        self.buffer_size = buffer_size
        self.max_buffers = max_buffers
        self._pool: List[bytearray] = []
        self._lock = threading.Lock()
        self._allocated = 0

    def acquire(self) -> bytearray:
        """Get a buffer from the pool (or allocate a new one)."""
        with self._lock:
            if self._pool:
                return self._pool.pop()
            self._allocated += 1
        return bytearray(self.buffer_size)

    def release(self, buf: bytearray) -> None:
        """Return a buffer to the pool for reuse."""
        with self._lock:
            if len(self._pool) < self.max_buffers:
                self._pool.append(buf)
            else:
                self._allocated -= 1

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "pooled": len(self._pool),
                "allocated": self._allocated,
                "buffer_size": self.buffer_size,
                "max_buffers": self.max_buffers,
            }

    def clear(self) -> None:
        """Release all pooled buffers."""
        with self._lock:
            self._pool.clear()
