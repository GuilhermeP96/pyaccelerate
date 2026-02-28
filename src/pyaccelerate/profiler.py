"""
pyaccelerate.profiler — Decorator-based profiling & timing utilities.

Provides zero-config decorators and context managers for measuring
execution time, memory usage, and call frequency:

  - ``@timed``          — log wall-clock time of each call
  - ``@profile_memory`` — log peak RSS delta after each call
  - ``Timer``           — context manager for scoped timing
  - ``Tracker``         — accumulates stats across many calls

Thread-safe. All output goes through ``logging`` (no print statements).
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional, TypeVar

log = logging.getLogger("pyaccelerate.profiler")

F = TypeVar("F", bound=Callable[..., Any])


# ═══════════════════════════════════════════════════════════════════════════
#  @timed decorator
# ═══════════════════════════════════════════════════════════════════════════

def timed(
    label: Optional[str] = None,
    level: int = logging.DEBUG,
) -> Callable[[F], F]:
    """Decorator that logs wall-clock time for each call.

    Usage::

        @timed()
        def process_data(data):
            ...

        @timed(label="heavy-op", level=logging.INFO)
        def heavy_operation():
            ...
    """

    def decorator(fn: F) -> F:
        name = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                log.log(level, "[%s] %.4f s", name, elapsed)

        return wrapper  # type: ignore[return-value]

    return decorator


# ═══════════════════════════════════════════════════════════════════════════
#  @profile_memory decorator
# ═══════════════════════════════════════════════════════════════════════════

def profile_memory(
    label: Optional[str] = None,
    level: int = logging.DEBUG,
) -> Callable[[F], F]:
    """Decorator that logs peak RSS delta after each call.

    Requires ``psutil``. If not installed, falls back to wall-clock timing only.
    """

    def decorator(fn: F) -> F:
        name = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rss_before = _get_rss_mb()
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                rss_after = _get_rss_mb()
                delta = rss_after - rss_before if rss_before >= 0 else 0
                log.log(
                    level,
                    "[%s] %.4f s | RSS delta: %+.1f MB (now %.1f MB)",
                    name, elapsed, delta, rss_after,
                )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _get_rss_mb() -> float:
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


# ═══════════════════════════════════════════════════════════════════════════
#  Timer context manager
# ═══════════════════════════════════════════════════════════════════════════

class Timer:
    """Scoped wall-clock timer, usable as a context manager.

    Usage::

        with Timer("load data") as t:
            data = load_big_file()
        print(f"took {t.elapsed:.3f} s")
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._t0
        if self.label:
            log.debug("[Timer:%s] %.4f s", self.label, self.elapsed)

    def __repr__(self) -> str:
        return f"Timer({self.label!r}, elapsed={self.elapsed:.4f})"


@contextmanager
def timer(label: str = "") -> Generator[Timer, None, None]:
    """Functional-style context manager wrapping ``Timer``."""
    t = Timer(label)
    with t:
        yield t


# ═══════════════════════════════════════════════════════════════════════════
#  Tracker — accumulates stats across many calls
# ═══════════════════════════════════════════════════════════════════════════

class Tracker:
    """Thread-safe statistics accumulator for repeated operations.

    Usage::

        tracker = Tracker("db_query")

        for batch in batches:
            with tracker.measure():
                run_query(batch)

        print(tracker.summary())
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._lock = threading.Lock()
        self._times: list[float] = []

    def record(self, elapsed: float) -> None:
        """Manually record a timing measurement."""
        with self._lock:
            self._times.append(elapsed)

    @contextmanager
    def measure(self) -> Generator[None, None, None]:
        """Context manager that auto-records elapsed time."""
        t0 = time.perf_counter()
        yield
        self.record(time.perf_counter() - t0)

    @property
    def count(self) -> int:
        return len(self._times)

    @property
    def total(self) -> float:
        return sum(self._times) if self._times else 0.0

    @property
    def mean(self) -> float:
        return self.total / len(self._times) if self._times else 0.0

    @property
    def min_time(self) -> float:
        return min(self._times) if self._times else 0.0

    @property
    def max_time(self) -> float:
        return max(self._times) if self._times else 0.0

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    def _percentile(self, pct: int) -> float:
        if not self._times:
            return 0.0
        sorted_t = sorted(self._times)
        idx = int(len(sorted_t) * pct / 100)
        idx = min(idx, len(sorted_t) - 1)
        return sorted_t[idx]

    def summary(self) -> str:
        """Human-readable statistics summary."""
        if not self._times:
            return f"Tracker({self.name}): no data"
        return (
            f"Tracker({self.name}): {self.count} calls | "
            f"total={self.total:.4f}s | mean={self.mean:.4f}s | "
            f"min={self.min_time:.4f}s | max={self.max_time:.4f}s | "
            f"p50={self.p50:.4f}s | p95={self.p95:.4f}s | p99={self.p99:.4f}s"
        )

    def stats_dict(self) -> Dict[str, float]:
        return {
            "count": float(self.count),
            "total_s": self.total,
            "mean_s": self.mean,
            "min_s": self.min_time,
            "max_s": self.max_time,
            "p50_s": self.p50,
            "p95_s": self.p95,
            "p99_s": self.p99,
        }

    def reset(self) -> None:
        with self._lock:
            self._times.clear()
