"""
pyaccelerate.adaptive — Adaptive scheduler that reacts to runtime pressure.

Continuously monitors **latency**, **CPU pressure** and **memory pressure**
and dynamically adjusts the work-stealing scheduler's behaviour:

- **Scale up** workers when latency is low and CPU headroom exists.
- **Scale down** when CPU utilisation exceeds a threshold or memory is tight.
- **Adjust steal batch size** to match current contention level.
- **Cooldown** window prevents oscillation (like a Kubernetes HPA).

Integrates with:
- ``pyaccelerate.work_stealing.WorkStealingScheduler``
- ``pyaccelerate.memory`` (pressure detection)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, TypeVar

from pyaccelerate.memory import Pressure, get_pressure

log = logging.getLogger("pyaccelerate.adaptive")

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════
#  Latency tracker (lock-free ring buffer)
# ═══════════════════════════════════════════════════════════════════════════

class LatencyTracker:
    """Fixed-size ring buffer that tracks task-completion latency in µs.

    All methods are safe to call from multiple threads (GIL-protected writes
    to the list + integer index).
    """

    __slots__ = ("_buf", "_size", "_pos", "_count")

    def __init__(self, window: int = 1024) -> None:
        self._size = window
        self._buf: list[float] = [0.0] * window
        self._pos = 0
        self._count = 0

    def record(self, latency_us: float) -> None:
        idx = self._pos % self._size
        self._buf[idx] = latency_us
        self._pos += 1
        if self._count < self._size:
            self._count += 1

    def percentile(self, p: float) -> float:
        """Return the *p*-th percentile (0-100) of recorded latencies."""
        if self._count == 0:
            return 0.0
        data = sorted(self._buf[: self._count])
        k = max(0, min(int(p / 100 * self._count) - 1, self._count - 1))
        return data[k]

    @property
    def p50(self) -> float:
        return self.percentile(50)

    @property
    def p95(self) -> float:
        return self.percentile(95)

    @property
    def p99(self) -> float:
        return self.percentile(99)

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._pos = 0
        self._count = 0


# ═══════════════════════════════════════════════════════════════════════════
#  CPU pressure helper
# ═══════════════════════════════════════════════════════════════════════════

def get_cpu_pressure() -> float:
    """Return CPU utilisation percentage (0-100).

    Falls back to ``os.getloadavg`` on POSIX or returns 50 if unavailable.
    """
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.cpu_percent(interval=0)
    except ImportError:
        pass
    try:
        load1, _, _ = os.getloadavg()
        cores = os.cpu_count() or 1
        return min(100.0, load1 / cores * 100)
    except (OSError, AttributeError):
        return 50.0


# ═══════════════════════════════════════════════════════════════════════════
#  Adaptive policy
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AdaptiveConfig:
    """Tuning knobs for the adaptive scheduler."""

    # ── Worker scaling ────────────────────────────────────────────────
    min_workers: int = 2
    max_workers: int = 0  # 0 → cpu_count × 2
    scale_up_cpu_threshold: float = 70.0      # CPU% below which we may add workers
    scale_down_cpu_threshold: float = 90.0    # CPU% above which we shed workers
    scale_up_latency_p95_us: float = 10_000   # scale up if P95 > this
    scale_down_latency_p95_us: float = 500    # scale down if P95 < this (idle)
    cooldown_seconds: float = 2.0             # min time between adjustments

    # ── Steal batch tuning ────────────────────────────────────────────
    min_steal_batch: int = 1
    max_steal_batch: int = 16

    # ── Monitor cadence ───────────────────────────────────────────────
    poll_interval_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.max_workers <= 0:
            self.max_workers = (os.cpu_count() or 4) * 2


@dataclass
class AdaptiveSnapshot:
    """Point-in-time metric snapshot exposed by the adaptive scheduler."""
    workers: int = 0
    steal_batch_size: int = 4
    cpu_pct: float = 0.0
    memory_pressure: str = "LOW"
    latency_p50_us: float = 0.0
    latency_p95_us: float = 0.0
    latency_p99_us: float = 0.0
    total_adjustments: int = 0
    last_action: str = "none"


class AdaptiveScheduler:
    """Wraps a ``WorkStealingScheduler`` and dynamically tunes it.

    Usage::

        from pyaccelerate.adaptive import AdaptiveScheduler

        sched = AdaptiveScheduler()
        fut = sched.submit(my_func, arg1)
        results = sched.map(fn, [(a,), (b,)])
        sched.shutdown()
    """

    def __init__(
        self,
        config: Optional[AdaptiveConfig] = None,
        num_workers: int = 0,
    ) -> None:
        from pyaccelerate.work_stealing import WorkStealingScheduler

        self._cfg = config or AdaptiveConfig()
        initial_workers = num_workers or max(
            self._cfg.min_workers,
            (os.cpu_count() or 4),
        )
        initial_workers = min(initial_workers, self._cfg.max_workers)

        self._scheduler = WorkStealingScheduler(
            num_workers=initial_workers,
            steal_batch_size=4,
        )
        self._latency = LatencyTracker()
        self._total_adjustments = 0
        self._last_adjust_time = 0.0
        self._last_action = "init"
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler and the adaptive monitor loop."""
        self._scheduler.start()
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="pyacc-adaptive",
            daemon=True,
        )
        self._monitor_thread.start()
        log.info("Adaptive scheduler started (%d workers)", self._scheduler._num_workers)

    def shutdown(self, wait: bool = True) -> None:
        """Stop monitor and underlying scheduler."""
        self._running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3.0)
        self._scheduler.shutdown(wait=wait)
        log.info("Adaptive scheduler shut down")

    def _ensure_started(self) -> None:
        if not self._running:
            self.start()

    def __enter__(self) -> "AdaptiveScheduler":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # ── Submit ───────────────────────────────────────────────────────

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Any:
        """Submit a task (auto-starts if needed)."""
        self._ensure_started()
        wrapped = self._wrap(fn)
        return self._scheduler.submit(wrapped, *args, **kwargs)

    def map(
        self,
        fn: Callable[..., T],
        items: Any,
        timeout: Optional[float] = None,
    ) -> List[T]:
        """Map a function over items with adaptive scheduling."""
        self._ensure_started()
        wrapped = self._wrap(fn)
        return self._scheduler.map(wrapped, items, timeout=timeout)

    def _wrap(self, fn: Callable[..., T]) -> Callable[..., T]:
        """Wrap *fn* to record execution latency."""
        tracker = self._latency

        def _instrumented(*args: Any, **kwargs: Any) -> T:
            t0 = time.monotonic_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_us = (time.monotonic_ns() - t0) / 1_000
                tracker.record(elapsed_us)

        return _instrumented  # type: ignore[return-value]

    # ── Monitor loop ─────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        cfg = self._cfg
        while self._running:
            time.sleep(cfg.poll_interval_seconds)
            if not self._running:
                break
            try:
                self._adapt()
            except Exception:
                log.exception("Adaptive monitor error")

    def _adapt(self) -> None:
        cfg = self._cfg
        now = time.monotonic()
        if now - self._last_adjust_time < cfg.cooldown_seconds:
            return

        cpu = get_cpu_pressure()
        mem = get_pressure()
        p95 = self._latency.p95

        current_workers = self._scheduler._num_workers
        new_workers = current_workers
        action = "hold"

        # ── Memory pressure override (highest priority) ──────────────
        if mem in (Pressure.HIGH, Pressure.CRITICAL):
            target = max(cfg.min_workers, current_workers // 2)
            if target < current_workers:
                new_workers = target
                action = f"scale_down(mem={mem.name})"

        # ── CPU pressure ──────────────────────────────────────────────
        elif cpu > cfg.scale_down_cpu_threshold and current_workers > cfg.min_workers:
            new_workers = max(cfg.min_workers, current_workers - 1)
            action = f"scale_down(cpu={cpu:.0f}%)"

        # ── Latency-driven scale-up ──────────────────────────────────
        elif (
            p95 > cfg.scale_up_latency_p95_us
            and cpu < cfg.scale_up_cpu_threshold
            and mem == Pressure.LOW
            and current_workers < cfg.max_workers
        ):
            new_workers = min(cfg.max_workers, current_workers + 1)
            action = f"scale_up(p95={p95:.0f}µs)"

        # ── Latency-driven scale-down (over-provisioned) ─────────────
        elif (
            p95 < cfg.scale_down_latency_p95_us
            and current_workers > cfg.min_workers
            and self._latency.count > 100
        ):
            new_workers = max(cfg.min_workers, current_workers - 1)
            action = f"scale_down(idle,p95={p95:.0f}µs)"

        # ── Apply worker change ──────────────────────────────────────
        if new_workers != current_workers:
            self._resize_workers(new_workers)
            self._total_adjustments += 1
            self._last_adjust_time = now
            self._last_action = action
            log.info("Adaptive: %s → %d workers", action, new_workers)

        # ── Steal batch tuning ───────────────────────────────────────
        self._tune_steal_batch(cpu)

    def _resize_workers(self, target: int) -> None:
        """Resize the underlying scheduler by replacing it.

        The old scheduler is shut down cleanly and a new one spins up with
        the desired worker count.  Pending futures that were already submitted
        will complete on the old scheduler.
        """
        from pyaccelerate.work_stealing import WorkStealingScheduler

        old = self._scheduler
        self._scheduler = WorkStealingScheduler(
            num_workers=target,
            steal_batch_size=old._steal_batch_size,
        )
        self._scheduler.start()
        # Allow old tasks to drain (best-effort)
        old.shutdown(wait=False)

    def _tune_steal_batch(self, cpu: float) -> None:
        """Adjust steal batch size based on CPU load."""
        cfg = self._cfg
        if cpu > 80:
            # High contention — smaller batches reduce overhead
            new_batch = max(cfg.min_steal_batch, self._scheduler._steal_batch_size - 1)
        elif cpu < 40:
            # Low contention — bigger batches reduce steal frequency
            new_batch = min(cfg.max_steal_batch, self._scheduler._steal_batch_size + 1)
        else:
            return
        self._scheduler._steal_batch_size = new_batch

    # ── Stats ────────────────────────────────────────────────────────

    def snapshot(self) -> AdaptiveSnapshot:
        """Return current adaptive scheduler metrics."""
        return AdaptiveSnapshot(
            workers=self._scheduler._num_workers,
            steal_batch_size=self._scheduler._steal_batch_size,
            cpu_pct=get_cpu_pressure(),
            memory_pressure=get_pressure().name,
            latency_p50_us=self._latency.p50,
            latency_p95_us=self._latency.p95,
            latency_p99_us=self._latency.p99,
            total_adjustments=self._total_adjustments,
            last_action=self._last_action,
        )

    @property
    def scheduler(self) -> Any:
        """Underlying ``WorkStealingScheduler``."""
        return self._scheduler
