"""
pyaccelerate.work_stealing — Work-stealing scheduler inspired by Tokio, Go & ForkJoinPool.

Architecture
------------
Each worker thread owns a **local WorkDeque** (Chase-Lev style):

1. **Pop** from own deque (LIFO — cache-locality, like Tokio's LIFO slot).
2. **Drain** the global injection queue (round-robin across workers).
3. **Steal** from a random victim's deque (FIFO — fairness, like ForkJoinPool).
4. **Park** with exponential back-off when there is no work.

The scheduler maintains a notification mechanism so that parked workers are
woken immediately when new work arrives (mirroring Go's ``runtime.ready``).

Thread-safety: all public methods are safe to call from any thread.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

from pyaccelerate.lockfree_queue import MPMCQueue, WorkDeque

log = logging.getLogger("pyaccelerate.work_stealing")

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════
#  Task wrapper
# ═══════════════════════════════════════════════════════════════════════════

class _Task:
    """Lightweight wrapper tying a callable to its ``Future``."""

    __slots__ = ("fn", "args", "kwargs", "future", "submit_ns")

    def __init__(
        self,
        fn: Callable[..., Any],
        args: tuple,
        kwargs: dict[str, Any],
        future: Future,  # type: ignore[type-arg]
    ) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.future = future
        self.submit_ns: int = time.monotonic_ns()


# ═══════════════════════════════════════════════════════════════════════════
#  Worker
# ═══════════════════════════════════════════════════════════════════════════

class _Worker:
    """A single worker thread bound to its own deque."""

    __slots__ = (
        "index",
        "deque",
        "thread",
        "_scheduler",
        "_running",
        "completed",
        "stolen",
        "total_latency_ns",
    )

    def __init__(self, index: int, scheduler: "WorkStealingScheduler") -> None:
        self.index = index
        self.deque: WorkDeque[_Task] = WorkDeque()
        self._scheduler = scheduler
        self._running = False
        self.completed: int = 0
        self.stolen: int = 0
        self.total_latency_ns: int = 0
        self.thread = threading.Thread(
            target=self._run,
            name=f"pyacc-ws-{index}",
            daemon=True,
        )

    def start(self) -> None:
        self._running = True
        self.thread.start()

    def stop(self) -> None:
        self._running = False

    # ── Main loop ────────────────────────────────────────────────────

    def _run(self) -> None:
        backoff_us = 1
        max_backoff_us = 5_000  # 5 ms

        while self._running:
            task = self._find_task()
            if task is not None:
                backoff_us = 1
                self._execute(task)
            else:
                # Exponential back-off park (like Go runtime)
                self._scheduler._global_queue.wait(backoff_us / 1_000_000)
                backoff_us = min(backoff_us * 2, max_backoff_us)

    def _find_task(self) -> Optional[_Task]:
        """Work-stealing search order: local → global → random victim."""
        # 1. Own deque (LIFO — cache-friendly, like Tokio's LIFO slot)
        task = self.deque.pop()
        if task is not None:
            return task

        # 2. Global injection queue
        task = self._scheduler._global_queue.get()
        if task is not None:
            # While we're at it, drain a few more into our local queue
            batch = self._scheduler._global_queue.get_batch(
                self._scheduler._steal_batch_size,
            )
            for t in batch:
                self.deque.push(t)
            return task

        # 3. Steal from a random victim (FIFO — fairness)
        workers = self._scheduler._workers
        n = len(workers)
        if n <= 1:
            return None

        start = random.randint(0, n - 1)
        for i in range(n):
            victim_idx = (start + i) % n
            if victim_idx == self.index:
                continue
            victim = workers[victim_idx]
            stolen = victim.deque.steal_batch(
                self._scheduler._steal_batch_size,
            )
            if stolen:
                self.stolen += len(stolen)
                task = stolen[0]
                for t in stolen[1:]:
                    self.deque.push(t)
                return task

        return None

    def _execute(self, task: _Task) -> None:
        """Execute a task and resolve its future."""
        if task.future.cancelled():
            return
        try:
            result = task.fn(*task.args, **task.kwargs)
            task.future.set_result(result)
        except BaseException as exc:
            task.future.set_exception(exc)
        finally:
            self.completed += 1
            self.total_latency_ns += time.monotonic_ns() - task.submit_ns


# ═══════════════════════════════════════════════════════════════════════════
#  Scheduler
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SchedulerStats:
    """Snapshot of scheduler metrics."""
    worker_count: int = 0
    total_submitted: int = 0
    total_completed: int = 0
    total_stolen: int = 0
    global_queue_depth: int = 0
    avg_latency_us: float = 0.0
    per_worker: List[Dict[str, int]] = field(default_factory=list)


class WorkStealingScheduler:
    """High-performance work-stealing scheduler.

    Parameters
    ----------
    num_workers : int
        Number of worker threads.  Defaults to ``os.cpu_count() or 4``.
    steal_batch_size : int
        Max items stolen per steal attempt.  Defaults to 4.
    global_queue_size : int
        Capacity of the global injection queue (0 = unbounded).
    """

    def __init__(
        self,
        num_workers: int = 0,
        *,
        steal_batch_size: int = 4,
        global_queue_size: int = 0,
    ) -> None:
        if num_workers <= 0:
            num_workers = os.cpu_count() or 4
        self._num_workers = num_workers
        self._steal_batch_size = steal_batch_size
        self._global_queue: MPMCQueue[_Task] = MPMCQueue(global_queue_size)
        self._workers: List[_Worker] = []
        self._submit_counter = 0
        self._lock = threading.Lock()
        self._started = False
        self._shutdown = False

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn all worker threads."""
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._workers = [
                _Worker(i, self) for i in range(self._num_workers)
            ]
            for w in self._workers:
                w.start()
            self._started = True
            log.info(
                "Work-stealing scheduler started: %d workers, steal_batch=%d",
                self._num_workers,
                self._steal_batch_size,
            )

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Stop all worker threads."""
        if self._shutdown:
            return
        self._shutdown = True
        for w in self._workers:
            w.stop()
        # Wake parked workers so they can exit
        self._global_queue.signal()
        if wait:
            deadline = time.monotonic() + timeout
            for w in self._workers:
                remaining = max(0.01, deadline - time.monotonic())
                w.thread.join(timeout=remaining)
        self._started = False
        log.info("Work-stealing scheduler shut down")

    def _ensure_started(self) -> None:
        if not self._started:
            self.start()

    # ── Context manager ──────────────────────────────────────────────

    def __enter__(self) -> "WorkStealingScheduler":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # ── Submit ───────────────────────────────────────────────────────

    def submit(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> "Future[T]":
        """Submit a task and return a ``Future``."""
        self._ensure_started()
        future: Future[T] = Future()
        task = _Task(fn, args, kwargs, future)  # type: ignore[arg-type]

        # Round-robin assignment to workers' local queues (like Go's P assignment)
        with self._lock:
            idx = self._submit_counter % self._num_workers
            self._submit_counter += 1

        self._workers[idx].deque.push(task)
        # Wake any parked workers
        self._global_queue.signal()
        return future

    def submit_to_global(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> "Future[T]":
        """Submit directly to the global queue (stolen by any idle worker)."""
        self._ensure_started()
        future: Future[T] = Future()
        task = _Task(fn, args, kwargs, future)  # type: ignore[arg-type]
        self._global_queue.put(task)
        return future

    def map(
        self,
        fn: Callable[..., T],
        items: Sequence[tuple],
        timeout: Optional[float] = None,
    ) -> List[T]:
        """Submit all items and collect results in order."""
        futures = [self.submit(fn, *item) for item in items]
        return [f.result(timeout=timeout) for f in futures]

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> SchedulerStats:
        """Return a snapshot of scheduler metrics."""
        total_completed = sum(w.completed for w in self._workers)
        total_stolen = sum(w.stolen for w in self._workers)
        total_latency = sum(w.total_latency_ns for w in self._workers)
        avg_us = (total_latency / total_completed / 1000) if total_completed else 0.0

        return SchedulerStats(
            worker_count=self._num_workers,
            total_submitted=self._submit_counter,
            total_completed=total_completed,
            total_stolen=total_stolen,
            global_queue_depth=len(self._global_queue),
            avg_latency_us=avg_us,
            per_worker=[
                {
                    "index": w.index,
                    "completed": w.completed,
                    "stolen": w.stolen,
                    "queue_depth": len(w.deque),
                }
                for w in self._workers
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Module-level singleton
# ═══════════════════════════════════════════════════════════════════════════

_default_scheduler: Optional[WorkStealingScheduler] = None
_default_lock = threading.Lock()


def get_scheduler(num_workers: int = 0) -> WorkStealingScheduler:
    """Return (and lazily create) the default work-stealing scheduler."""
    global _default_scheduler
    if _default_scheduler is None:
        with _default_lock:
            if _default_scheduler is None:
                _default_scheduler = WorkStealingScheduler(num_workers=num_workers)
                _default_scheduler.start()
    return _default_scheduler


def shutdown_scheduler(wait: bool = True) -> None:
    """Shut down the default scheduler."""
    global _default_scheduler
    if _default_scheduler is not None:
        _default_scheduler.shutdown(wait=wait)
        _default_scheduler = None


def ws_submit(fn: Callable[..., T], *args: Any, **kwargs: Any) -> "Future[T]":
    """Convenience: submit via the default work-stealing scheduler."""
    return get_scheduler().submit(fn, *args, **kwargs)


def ws_map(
    fn: Callable[..., T],
    items: Sequence[tuple],
    timeout: Optional[float] = None,
) -> List[T]:
    """Convenience: map via the default work-stealing scheduler."""
    return get_scheduler().map(fn, items, timeout=timeout)
