"""
pyaccelerate.pipeline — Typed pipeline stages with backpressure.

Build data-processing pipelines where each stage runs in parallel and feeds
the next through bounded queues (backpressure prevents memory blow-up).

Usage::

    from pyaccelerate.pipeline import Pipeline, Stage

    pipeline = Pipeline([
        Stage("download", download_fn, concurrency=8),
        Stage("transform", transform_fn, concurrency=4),
        Stage("save", save_fn, concurrency=2),
    ])

    results = pipeline.run(input_items)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

from pyaccelerate.profiler import Timer

log = logging.getLogger("pyaccelerate.pipeline")

T = TypeVar("T")

_SENTINEL = object()


@dataclass
class StageStats:
    """Runtime statistics for a single pipeline stage."""
    name: str = ""
    items_in: int = 0
    items_out: int = 0
    errors: int = 0
    elapsed_s: float = 0.0
    avg_item_ms: float = 0.0


@dataclass
class PipelineResult:
    """Result of a full pipeline execution."""
    outputs: List[Any] = field(default_factory=list)
    stages: List[StageStats] = field(default_factory=list)
    total_elapsed_s: float = 0.0
    total_errors: int = 0

    def summary(self) -> str:
        lines = [f"Pipeline completed in {self.total_elapsed_s:.2f}s"]
        for s in self.stages:
            lines.append(
                f"  {s.name}: {s.items_out}/{s.items_in} items, "
                f"{s.errors} errors, {s.elapsed_s:.2f}s "
                f"(avg {s.avg_item_ms:.1f}ms/item)"
            )
        if self.total_errors:
            lines.append(f"  Total errors: {self.total_errors}")
        return "\n".join(lines)


class Stage:
    """A single processing stage in a pipeline.

    Parameters
    ----------
    name
        Human-readable stage name.
    fn
        Processing function. Receives one item, returns one result.
    concurrency
        Number of parallel workers for this stage.
    queue_size
        Max items buffered between this stage and the next (backpressure).
    on_error
        Error handler. If None, errors are logged and the item is dropped.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[Any], Any],
        concurrency: int = 4,
        queue_size: int = 100,
        on_error: Optional[Callable[[Exception, Any], Any]] = None,
    ):
        self.name = name
        self.fn = fn
        self.concurrency = max(1, concurrency)
        self.queue_size = max(1, queue_size)
        self.on_error = on_error


class Pipeline:
    """Multi-stage parallel processing pipeline with backpressure.

    Each stage runs ``concurrency`` worker threads. Items flow through
    bounded queues between stages, providing natural backpressure.

    Usage::

        pipeline = Pipeline([
            Stage("extract", extract_fn, concurrency=2),
            Stage("transform", transform_fn, concurrency=8),
            Stage("load", load_fn, concurrency=2),
        ])
        result = pipeline.run(input_items)
        print(result.summary())
    """

    def __init__(self, stages: Sequence[Stage]):
        if not stages:
            raise ValueError("Pipeline requires at least one stage")
        self._stages = list(stages)

    def run(self, items: Sequence[Any]) -> PipelineResult:
        """Execute the pipeline with the given input items.

        Returns a :class:`PipelineResult` with outputs and per-stage stats.
        """
        n_stages = len(self._stages)
        # Create inter-stage queues (input_queue → stage → output_queue)
        queues: List[queue.Queue] = []
        for i in range(n_stages + 1):
            size = self._stages[min(i, n_stages - 1)].queue_size if i < n_stages else 0
            queues.append(queue.Queue(maxsize=max(size, 1)))

        stage_stats: List[StageStats] = []
        all_threads: List[List[threading.Thread]] = []

        t0 = time.perf_counter()

        # Set up worker threads for each stage
        for idx, stage in enumerate(self._stages):
            in_q = queues[idx]
            out_q = queues[idx + 1]
            stats = StageStats(name=stage.name)
            stage_stats.append(stats)

            threads = []
            for w in range(stage.concurrency):
                t = threading.Thread(
                    target=self._worker,
                    args=(stage, in_q, out_q, stats),
                    name=f"pipeline-{stage.name}-{w}",
                    daemon=True,
                )
                threads.append(t)
                t.start()
            all_threads.append(threads)

        # Feed input items into the first queue
        for item in items:
            queues[0].put(item)
            stage_stats[0].items_in += 1

        # Send sentinel for each worker of the first stage
        for _ in range(self._stages[0].concurrency):
            queues[0].put(_SENTINEL)

        # Wait for all stages to complete
        for idx, threads in enumerate(all_threads):
            for t in threads:
                t.join()
            # Send sentinels to the next stage's workers
            if idx + 1 < n_stages:
                for _ in range(self._stages[idx + 1].concurrency):
                    queues[idx + 1].put(_SENTINEL)

        # Collect outputs from the last queue
        outputs = []
        last_q = queues[-1]
        while not last_q.empty():
            item = last_q.get_nowait()
            if item is not _SENTINEL:
                outputs.append(item)

        total_elapsed = time.perf_counter() - t0
        total_errors = sum(s.errors for s in stage_stats)

        # Compute per-stage timing
        for s in stage_stats:
            if s.items_out > 0 and s.elapsed_s > 0:
                s.avg_item_ms = (s.elapsed_s / s.items_out) * 1000

        result = PipelineResult(
            outputs=outputs,
            stages=stage_stats,
            total_elapsed_s=round(total_elapsed, 4),
            total_errors=total_errors,
        )
        log.info(result.summary())
        return result

    @staticmethod
    def _worker(
        stage: Stage,
        in_q: queue.Queue,
        out_q: queue.Queue,
        stats: StageStats,
    ) -> None:
        """Worker loop: pull from input queue, process, push to output queue."""
        while True:
            item = in_q.get()
            if item is _SENTINEL:
                break

            t0 = time.perf_counter()
            try:
                result = stage.fn(item)
                stats.elapsed_s += time.perf_counter() - t0
                stats.items_out += 1
                out_q.put(result)
            except Exception as exc:
                stats.elapsed_s += time.perf_counter() - t0
                stats.errors += 1
                if stage.on_error is not None:
                    try:
                        fallback = stage.on_error(exc, item)
                        if fallback is not None:
                            out_q.put(fallback)
                            stats.items_out += 1
                    except Exception:
                        log.warning("Stage '%s' error handler failed: %s", stage.name, exc)
                else:
                    log.warning("Stage '%s' failed on item: %s", stage.name, exc)
