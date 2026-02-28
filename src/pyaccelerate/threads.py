"""
pyaccelerate.threads — Virtual thread pool, sliding-window executor & async bridge.

Core concepts
-------------
- **Virtual thread pool**: A persistent, shared ``ThreadPoolExecutor`` sized for
  I/O-bound workloads (``logical_cores × 3``, capped at 32). Threads are named
  ``pyacc-vt-*`` for easy identification in profilers / debuggers.
- **Sliding-window executor**: ``run_parallel()`` submits tasks with a bounded
  concurrency window — as one finishes, the next is submitted, keeping the
  pipeline fed without overwhelming system resources.
- **Async bridge**: ``run_in_executor()`` / ``gather_parallel()`` integrate
  the pool with ``asyncio`` for hybrid sync+async codebases.
- **Process pool**: ``get_process_pool()`` for CPU-bound work that needs to
  bypass the GIL entirely.

Thread-safety: all public functions are safe to call from any thread.

Usage::

    from pyaccelerate.threads import get_pool, run_parallel, submit

    # Submit a single task
    fut = submit(my_func, arg1, arg2)

    # Run many tasks with bounded concurrency
    run_parallel(process_file, [(f,) for f in files], max_concurrent=8)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
)
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

log = logging.getLogger("pyaccelerate.threads")

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════
#  Pool sizing helpers
# ═══════════════════════════════════════════════════════════════════════════

def io_pool_size() -> int:
    """Optimal size for the shared I/O thread pool.

    I/O-bound threads spend >95% of time blocked (GIL released), so we can
    safely use ``cores × 3`` (capped at 32). Clamped on low-RAM hosts.
    """
    cores = os.cpu_count() or 4
    ram_gb = _get_ram_gb()
    size = min(cores * 3, 32)
    if ram_gb < 4:
        size = min(size, 12)
    return max(4, size)


def cpu_pool_size() -> int:
    """Optimal size for a CPU-bound process pool (= physical cores)."""
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.cpu_count(logical=False) or (os.cpu_count() or 4)
    except ImportError:
        return os.cpu_count() or 4


# ═══════════════════════════════════════════════════════════════════════════
#  Shared persistent I/O thread pool  ("virtual threads")
# ═══════════════════════════════════════════════════════════════════════════
_io_pool: Optional[ThreadPoolExecutor] = None
_io_pool_lock = threading.Lock()

_proc_pool: Optional[ProcessPoolExecutor] = None
_proc_pool_lock = threading.Lock()


def get_pool() -> ThreadPoolExecutor:
    """Return (and lazily create) the shared I/O virtual-thread pool.

    The pool is a module-level singleton — safe to call from any thread.
    Workers are named ``pyacc-vt-*`` for easy identification.
    """
    global _io_pool
    if _io_pool is None:
        with _io_pool_lock:
            if _io_pool is None:
                size = io_pool_size()
                _io_pool = ThreadPoolExecutor(
                    max_workers=size,
                    thread_name_prefix="pyacc-vt",
                )
                log.info("Virtual thread pool created: %d workers", size)
    return _io_pool


def get_process_pool() -> ProcessPoolExecutor:
    """Return (and lazily create) a shared ``ProcessPoolExecutor``.

    Sized for CPU-bound work (= physical core count). Use this when you
    need to bypass the GIL entirely.
    """
    global _proc_pool
    if _proc_pool is None:
        with _proc_pool_lock:
            if _proc_pool is None:
                size = cpu_pool_size()
                _proc_pool = ProcessPoolExecutor(max_workers=size)
                log.info("Process pool created: %d workers", size)
    return _proc_pool


def shutdown_pools(wait_for: bool = True) -> None:
    """Shut down all shared pools. Call during application shutdown."""
    global _io_pool, _proc_pool
    if _io_pool is not None:
        _io_pool.shutdown(wait=wait_for)
        _io_pool = None
    if _proc_pool is not None:
        _proc_pool.shutdown(wait=wait_for)
        _proc_pool = None
    log.info("All shared pools shut down")


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience submission
# ═══════════════════════════════════════════════════════════════════════════

def submit(fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
    """Submit a single task to the shared virtual-thread pool."""
    pool = get_pool()
    return pool.submit(fn, *args, **kwargs)


def submit_cpu(fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
    """Submit a single CPU-bound task to the process pool."""
    pool = get_process_pool()
    return pool.submit(fn, *args, **kwargs)


def map_parallel(
    fn: Callable[..., T],
    items: Sequence[tuple],
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
) -> List[T]:
    """Execute ``fn(*item)`` for each item and return results in order.

    Uses the shared pool. Raises the first exception encountered.
    """
    pool = get_pool()
    futures: List[Future[T]] = []
    for item in items:
        futures.append(pool.submit(fn, *item))

    results: List[T] = []
    for fut in futures:
        results.append(fut.result(timeout=timeout))
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Sliding-window parallel executor
# ═══════════════════════════════════════════════════════════════════════════

def run_parallel(
    fn: Callable[..., Any],
    items: Iterable[tuple],
    max_concurrent: int,
    *,
    on_done: Optional[Callable[[Future], None]] = None,
    on_error: Optional[Callable[[Exception, tuple], None]] = None,
) -> int:
    """Execute ``fn(*item)`` for each item with bounded concurrency.

    Uses a **sliding-window** strategy on the shared I/O pool: at most
    *max_concurrent* tasks run simultaneously. As one finishes, the next
    is submitted — keeping throughput high without overwhelming resources.

    Parameters
    ----------
    fn:
        Callable receiving unpacked elements of each item tuple.
    items:
        Iterable of argument tuples — ``fn(*item)`` for each.
    max_concurrent:
        Maximum number of tasks in-flight at once.
    on_done:
        Optional callback invoked with each completed ``Future``.
    on_error:
        Optional callback invoked when a task raises. If not provided,
        errors are logged and silently swallowed.

    Returns
    -------
    int
        Total number of completed tasks (success + failure).
    """
    pool = get_pool()
    active: Dict[Future, tuple] = {}
    it = iter(items)
    completed = 0

    # Fill initial window
    for _ in range(max_concurrent):
        try:
            args = next(it)
            active[pool.submit(fn, *args)] = args
        except StopIteration:
            break

    # Sliding window: as one completes, submit the next
    while active:
        done_set, _ = wait(active, return_when=FIRST_COMPLETED)
        for fut in done_set:
            task_args = active.pop(fut)
            completed += 1

            if on_done is not None:
                on_done(fut)
            else:
                try:
                    fut.result()
                except Exception as exc:
                    if on_error is not None:
                        on_error(exc, task_args)
                    else:
                        log.warning("Task %s failed: %s", task_args, exc)

            try:
                args = next(it)
                active[pool.submit(fn, *args)] = args
            except StopIteration:
                pass

    return completed


def run_parallel_collect(
    fn: Callable[..., T],
    items: Sequence[tuple],
    max_concurrent: int,
) -> List[Tuple[tuple, T | Exception]]:
    """Like ``run_parallel`` but collects all results (or exceptions).

    Returns a list of ``(args, result_or_exception)`` in completion order.
    """
    results: List[Tuple[tuple, T | Exception]] = []
    lock = threading.Lock()

    def _collect(fut: Future) -> None:
        nonlocal results
        # We need the args — traverse through active map
        try:
            val = fut.result()
        except Exception as exc:
            val = exc  # type: ignore[assignment]
        with lock:
            results.append(((), val))

    # Use direct approach for ordered results
    pool = get_pool()
    futures_map: Dict[Future, tuple] = {}
    for item in items:
        futures_map[pool.submit(fn, *item)] = item

    for fut in as_completed(futures_map):
        args = futures_map[fut]
        try:
            val = fut.result()
            results.append((args, val))
        except Exception as exc:
            results.append((args, exc))

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Async bridge
# ═══════════════════════════════════════════════════════════════════════════

async def run_in_executor(
    fn: Callable[..., T],
    *args: Any,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> T:
    """Run a blocking function on the shared pool from an async context.

    Example::

        result = await run_in_executor(heavy_io_func, arg1, arg2)
    """
    if loop is None:
        loop = asyncio.get_running_loop()
    pool = get_pool()
    return await loop.run_in_executor(pool, fn, *args)


async def gather_parallel(
    fn: Callable[..., T],
    items: Sequence[tuple],
    max_concurrent: int = 0,
) -> List[T]:
    """Async-friendly parallel dispatch returning ordered results.

    If *max_concurrent* is 0, all items are dispatched at once. Otherwise,
    a ``Semaphore`` limits concurrency.
    """
    loop = asyncio.get_running_loop()
    pool = get_pool()

    if max_concurrent <= 0:
        coros = [loop.run_in_executor(pool, fn, *item) for item in items]
        return list(await asyncio.gather(*coros))

    sem = asyncio.Semaphore(max_concurrent)

    async def _limited(item: tuple) -> T:
        async with sem:
            return await loop.run_in_executor(pool, fn, *item)

    coros = [_limited(item) for item in items]
    return list(await asyncio.gather(*coros))


# ═══════════════════════════════════════════════════════════════════════════
#  Batch helper (tqdm-compatible progress)
# ═══════════════════════════════════════════════════════════════════════════

def batch_execute(
    fn: Callable[..., T],
    items: Sequence[tuple],
    max_concurrent: int = 0,
    desc: str = "Processing",
    show_progress: bool = True,
) -> List[T]:
    """Execute tasks with an optional tqdm progress bar.

    Parameters
    ----------
    max_concurrent:
        0 = use ``io_pool_size()``.
    show_progress:
        If True and ``tqdm`` is installed, show a progress bar.
    """
    if max_concurrent <= 0:
        max_concurrent = io_pool_size()

    pool = get_pool()
    total = len(items)
    results: List[Optional[T]] = [None] * total

    progress = None
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
            progress = tqdm(total=total, desc=desc, unit="item")
        except ImportError:
            pass

    futures_map: Dict[Future, int] = {}
    for idx, item in enumerate(items):
        futures_map[pool.submit(fn, *item)] = idx

    for fut in as_completed(futures_map):
        idx = futures_map[fut]
        try:
            results[idx] = fut.result()
        except Exception as exc:
            log.warning("batch_execute item %d failed: %s", idx, exc)
            results[idx] = None  # type: ignore[assignment]
        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.close()

    return results  # type: ignore[return-value]


# ─────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────

def _get_ram_gb() -> float:
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return 8.0
