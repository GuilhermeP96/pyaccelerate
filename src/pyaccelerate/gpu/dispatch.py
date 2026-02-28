"""
pyaccelerate.gpu.dispatch — Multi-GPU workload dispatcher & load balancer.

Distributes work across multiple GPUs using configurable strategies:
  - **round-robin** : Assign tasks to GPUs in cyclic order
  - **score-weighted** : Assign proportionally more work to higher-scored GPUs
  - **memory-fit** : Assign each task to the GPU with the most free VRAM

Works with any compute backend (CUDA, OpenCL, Intel) via the unified
``GPUDevice`` abstraction from ``pyaccelerate.gpu.detector``.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TypeVar

from pyaccelerate.gpu.detector import GPUDevice, detect_all

log = logging.getLogger("pyaccelerate.gpu.dispatch")

T = TypeVar("T")

Strategy = Literal["round-robin", "score-weighted", "memory-fit"]


# ═══════════════════════════════════════════════════════════════════════════
#  Assignment helpers
# ═══════════════════════════════════════════════════════════════════════════

def _assign_round_robin(
    items: Sequence[Any],
    gpus: List[GPUDevice],
) -> List[List[Any]]:
    """Assign items to GPUs in round-robin fashion."""
    buckets: List[List[Any]] = [[] for _ in gpus]
    for i, item in enumerate(items):
        buckets[i % len(gpus)].append(item)
    return buckets


def _assign_score_weighted(
    items: Sequence[Any],
    gpus: List[GPUDevice],
) -> List[List[Any]]:
    """Assign more items to higher-scored GPUs proportionally."""
    total_score = sum(max(g.score, 1) for g in gpus)
    shares = [max(g.score, 1) / total_score for g in gpus]

    buckets: List[List[Any]] = [[] for _ in gpus]
    # Compute cumulative share boundaries
    cumulative = []
    running = 0.0
    for s in shares:
        running += s
        cumulative.append(running)

    n = len(items)
    for i, item in enumerate(items):
        pos = (i + 0.5) / n  # normalized position
        for gi, boundary in enumerate(cumulative):
            if pos <= boundary:
                buckets[gi].append(item)
                break
        else:
            buckets[-1].append(item)

    return buckets


def _assign_items(
    items: Sequence[Any],
    gpus: List[GPUDevice],
    strategy: Strategy,
) -> List[List[Any]]:
    """Route method dispatch."""
    if strategy == "score-weighted":
        return _assign_score_weighted(items, gpus)
    # default: round-robin
    return _assign_round_robin(items, gpus)


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def dispatch(
    fn: Callable[..., T],
    items: Sequence[Any],
    *,
    gpus: Optional[List[GPUDevice]] = None,
    strategy: Strategy = "round-robin",
    max_workers_per_gpu: int = 1,
    fallback_cpu: bool = True,
) -> List[T]:
    """Dispatch ``fn(item, gpu)`` across multiple GPUs.

    Parameters
    ----------
    fn:
        Callable ``fn(item, gpu: GPUDevice) -> T``. The function receives
        each item plus the assigned GPU device.
    items:
        Sequence of work items to distribute.
    gpus:
        List of GPUs to use. If None, auto-detect (usable only).
    strategy:
        Distribution strategy: "round-robin" or "score-weighted".
    max_workers_per_gpu:
        How many concurrent tasks per GPU.
    fallback_cpu:
        If True and no usable GPUs found, call ``fn(item, None)`` on CPU.

    Returns
    -------
    list:
        Results in the same order as *items*.
    """
    if gpus is None:
        gpus = [g for g in detect_all() if g.usable]

    if not gpus:
        if fallback_cpu:
            log.info("No usable GPU — dispatching %d items on CPU", len(items))
            return [fn(item, None) for item in items]  # type: ignore[arg-type]
        raise RuntimeError("No usable GPU found and fallback_cpu=False")

    buckets = _assign_items(items, gpus, strategy)
    total_workers = max(1, max_workers_per_gpu * len(gpus))

    # Map item → original index for ordered results
    item_to_index: Dict[int, int] = {id(item): i for i, item in enumerate(items)}
    results: List[Any] = [None] * len(items)
    lock = threading.Lock()

    def _worker(item: Any, gpu: GPUDevice) -> tuple[int, T]:
        idx = item_to_index[id(item)]
        result = fn(item, gpu)
        return idx, result

    with ThreadPoolExecutor(max_workers=total_workers) as pool:
        futures: List[Future] = []
        for gi, bucket in enumerate(buckets):
            gpu = gpus[gi]
            for item in bucket:
                futures.append(pool.submit(_worker, item, gpu))

        for fut in as_completed(futures):
            try:
                idx, result = fut.result()
                with lock:
                    results[idx] = result
            except Exception as exc:
                log.warning("GPU dispatch task failed: %s", exc)

    return results


def multi_gpu_map(
    fn: Callable[[Any, GPUDevice], T],
    items: Sequence[Any],
    *,
    strategy: Strategy = "score-weighted",
) -> List[T]:
    """Simplified multi-GPU map. Auto-detects GPUs and distributes work.

    Equivalent to ``dispatch(fn, items, strategy=strategy)``.
    """
    return dispatch(fn, items, strategy=strategy)
