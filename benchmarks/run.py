"""
PyAccelerate Benchmark Suite
=============================

Compares pyaccelerate against standard-library and popular alternatives
across IO-bound, CPU-bound, and mixed workloads.

Usage:
    python -m benchmarks.run            # full suite
    python -m benchmarks.run --io       # IO-bound only
    python -m benchmarks.run --cpu      # CPU-bound only
    python -m benchmarks.run --mixed    # mixed only
    python -m benchmarks.run --quick    # fewer iterations (CI-friendly)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import statistics
import struct
import sys
import textwrap
import time
import zlib
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the local src/ is importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pyaccelerate as pa  # noqa: E402
from pyaccelerate.threads import (  # noqa: E402
    get_pool,
    map_parallel,
    run_parallel,
)
from pyaccelerate.work_stealing import (  # noqa: E402
    WorkStealingScheduler,
    ws_map,
)

# ---------------------------------------------------------------------------
# Workload functions
# ---------------------------------------------------------------------------

# IO-bound: simulate network latency (sleep) — no real sockets needed
_IO_SLEEP = 0.02  # 20ms per "request"


def _io_task(idx: int) -> int:
    """Simulate an IO-bound task (HTTP call, DB query, file read)."""
    time.sleep(_IO_SLEEP)
    return idx


# IO-bound with variable latency (realistic: some fast, some slow)
import random as _random

_random.seed(42)
_VAR_LATENCIES = [_random.uniform(0.005, 0.08) for _ in range(5000)]


def _io_variable_task(idx: int) -> int:
    """Simulate IO with variable latency (5ms–80ms)."""
    time.sleep(_VAR_LATENCIES[idx % len(_VAR_LATENCIES)])
    return idx


# CPU-bound: SHA-256 hash chain
_HASH_ROUNDS = 800


def _cpu_task(idx: int) -> bytes:
    """SHA-256 hash chain — pure CPU, no GIL release."""
    data = struct.pack(">I", idx)
    for _ in range(_HASH_ROUNDS):
        data = hashlib.sha256(data).digest()
    return data


# CPU-bound: zlib compression
_COMPRESS_SIZE = 200_000  # bytes


def _compress_task(idx: int) -> int:
    """Compress random-ish data — exercises C extension (GIL released)."""
    blob = bytes(range(256)) * (_COMPRESS_SIZE // 256) + struct.pack(">I", idx)
    return len(zlib.compress(blob, 6))


# Mixed workload: IO + CPU interleaved
def _mixed_task(idx: int) -> int:
    """IO wait then CPU work."""
    time.sleep(_IO_SLEEP / 2)
    data = struct.pack(">I", idx)
    for _ in range(_HASH_ROUNDS // 2):
        data = hashlib.sha256(data).digest()
    return idx


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _time_it(label: str, fn, *args, **kwargs) -> tuple[float, object]:
    """Return (elapsed_seconds, result)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return elapsed, result


def _sequential(fn, items):
    return [fn(i) for i in items]


def _thread_pool_executor(fn, items, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(fn, i) for i in items]
        return [f.result() for f in futs]


def _process_pool_executor(fn, items, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(fn, i) for i in items]
        return [f.result() for f in futs]


def _pyacc_thread_pool(fn, items):
    """pyaccelerate ThreadPool (virtual-thread style)."""
    return map_parallel(fn, [(i,) for i in items])


def _pyacc_work_stealing(fn, items):
    """pyaccelerate WorkStealingScheduler."""
    return ws_map(fn, [(i,) for i in items])


def _pyacc_engine_batch(fn, items):
    """pyaccelerate Engine.batch()."""
    engine = pa.Engine(auto_threads=True)
    try:
        return engine.batch(fn, [(i,) for i in items], show_progress=False)
    finally:
        engine.shutdown()


def _pyacc_adaptive(fn, items):
    """pyaccelerate AdaptiveScheduler."""
    with pa.AdaptiveScheduler() as sched:
        return sched.map(fn, [(i,) for i in items])


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

class BenchmarkResult:
    __slots__ = ("category", "workload", "runner", "tasks", "elapsed", "throughput")

    def __init__(self, category: str, workload: str, runner: str,
                 tasks: int, elapsed: float):
        self.category = category
        self.workload = workload
        self.runner = runner
        self.tasks = tasks
        self.elapsed = elapsed
        self.throughput = tasks / elapsed if elapsed > 0 else float("inf")


def _run_bench(category: str, workload: str, fn, items,
               runners: dict, *, repeats: int = 3) -> list[BenchmarkResult]:
    results = []
    for label, runner_fn in runners.items():
        times = []
        for _ in range(repeats):
            elapsed, _ = _time_it(label, runner_fn, fn, items)
            times.append(elapsed)
        best = min(times)
        results.append(BenchmarkResult(category, workload, label, len(items), best))
    return results


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

def bench_io(n_tasks: int, repeats: int, workers: int) -> list[BenchmarkResult]:
    items = list(range(n_tasks))
    runners = {
        "Sequential":            lambda fn, it: _sequential(fn, it),
        "ThreadPoolExecutor":    lambda fn, it: _thread_pool_executor(fn, it, workers),
        "pyaccelerate.threads":  lambda fn, it: _pyacc_thread_pool(fn, it),
        "pyaccelerate.ws":       lambda fn, it: _pyacc_work_stealing(fn, it),
        "pyaccelerate.engine":   lambda fn, it: _pyacc_engine_batch(fn, it),
        "pyaccelerate.adaptive": lambda fn, it: _pyacc_adaptive(fn, it),
    }
    return _run_bench("IO-bound", f"sleep {_IO_SLEEP*1000:.0f}ms × {n_tasks}",
                       _io_task, items, runners, repeats=repeats)


def bench_io_variable(n_tasks: int, repeats: int, workers: int) -> list[BenchmarkResult]:
    """Variable-latency IO — where work-stealing shines."""
    items = list(range(n_tasks))
    runners = {
        "Sequential":            lambda fn, it: _sequential(fn, it),
        "ThreadPoolExecutor":    lambda fn, it: _thread_pool_executor(fn, it, workers),
        "pyaccelerate.threads":  lambda fn, it: _pyacc_thread_pool(fn, it),
        "pyaccelerate.ws":       lambda fn, it: _pyacc_work_stealing(fn, it),
        "pyaccelerate.adaptive": lambda fn, it: _pyacc_adaptive(fn, it),
    }
    return _run_bench("IO-bound", f"variable 5–80ms × {n_tasks}",
                       _io_variable_task, items, runners, repeats=repeats)


def bench_cpu_hash(n_tasks: int, repeats: int, workers: int) -> list[BenchmarkResult]:
    items = list(range(n_tasks))
    runners = {
        "Sequential":            lambda fn, it: _sequential(fn, it),
        "ThreadPoolExecutor":    lambda fn, it: _thread_pool_executor(fn, it, workers),
        "pyaccelerate.threads":  lambda fn, it: _pyacc_thread_pool(fn, it),
        "pyaccelerate.ws":       lambda fn, it: _pyacc_work_stealing(fn, it),
        "pyaccelerate.engine":   lambda fn, it: _pyacc_engine_batch(fn, it),
        "pyaccelerate.adaptive": lambda fn, it: _pyacc_adaptive(fn, it),
    }
    return _run_bench("CPU-bound", f"SHA-256 ×{_HASH_ROUNDS} chain × {n_tasks}",
                       _cpu_task, items, runners, repeats=repeats)


def bench_cpu_compress(n_tasks: int, repeats: int, workers: int) -> list[BenchmarkResult]:
    items = list(range(n_tasks))
    runners = {
        "Sequential":            lambda fn, it: _sequential(fn, it),
        "ThreadPoolExecutor":    lambda fn, it: _thread_pool_executor(fn, it, workers),
        "pyaccelerate.threads":  lambda fn, it: _pyacc_thread_pool(fn, it),
        "pyaccelerate.ws":       lambda fn, it: _pyacc_work_stealing(fn, it),
        "pyaccelerate.adaptive": lambda fn, it: _pyacc_adaptive(fn, it),
    }
    return _run_bench("CPU-bound", f"zlib compress {_COMPRESS_SIZE//1000}KB × {n_tasks}",
                       _compress_task, items, runners, repeats=repeats)


def bench_mixed(n_tasks: int, repeats: int, workers: int) -> list[BenchmarkResult]:
    items = list(range(n_tasks))
    runners = {
        "Sequential":            lambda fn, it: _sequential(fn, it),
        "ThreadPoolExecutor":    lambda fn, it: _thread_pool_executor(fn, it, workers),
        "pyaccelerate.threads":  lambda fn, it: _pyacc_thread_pool(fn, it),
        "pyaccelerate.ws":       lambda fn, it: _pyacc_work_stealing(fn, it),
        "pyaccelerate.engine":   lambda fn, it: _pyacc_engine_batch(fn, it),
        "pyaccelerate.adaptive": lambda fn, it: _pyacc_adaptive(fn, it),
    }
    return _run_bench("Mixed", f"IO({_IO_SLEEP*500:.0f}ms)+CPU(SHA×{_HASH_ROUNDS//2}) × {n_tasks}",
                       _mixed_task, items, runners, repeats=repeats)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_table(results: list[BenchmarkResult]) -> str:
    """Pretty-print results as aligned Markdown table."""
    if not results:
        return ""

    # Group by (category, workload)
    groups: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        key = f"{r.category}: {r.workload}"
        groups.setdefault(key, []).append(r)

    lines: list[str] = []
    for group_label, rows in groups.items():
        # sort by elapsed
        rows.sort(key=lambda r: r.elapsed)
        baseline = rows[-1].elapsed  # slowest = baseline (usually sequential)
        seq_time = next((r.elapsed for r in rows if r.runner == "Sequential"), baseline)

        lines.append(f"\n### {group_label}\n")
        lines.append("| Runner | Time (s) | Speedup | Tasks/sec |")
        lines.append("|--------|----------|---------|-----------|")
        for r in rows:
            speedup = seq_time / r.elapsed if r.elapsed > 0 else float("inf")
            lines.append(
                f"| {r.runner:<24s} | {r.elapsed:>8.3f} | {speedup:>6.1f}× | {r.throughput:>9.0f} |"
            )

    return "\n".join(lines)


def _format_summary_table(results: list[BenchmarkResult]) -> str:
    """One-line-per-workload summary for the README hero section."""
    groups: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        key = f"{r.category}: {r.workload}"
        groups.setdefault(key, []).append(r)

    lines = [
        "| Workload | Sequential | ThreadPool | pyaccelerate (best) | Speedup |",
        "|----------|------------|------------|---------------------|---------|",
    ]
    for group_label, rows in groups.items():
        seq = next((r for r in rows if r.runner == "Sequential"), None)
        tpe = next((r for r in rows if r.runner == "ThreadPoolExecutor"), None)
        pa_best = min(
            (r for r in rows if r.runner.startswith("pyaccelerate")),
            key=lambda r: r.elapsed,
            default=None,
        )
        if seq and pa_best:
            speedup = seq.elapsed / pa_best.elapsed
            short_label = group_label.split(": ", 1)[-1]
            tpe_col = f"{tpe.elapsed:.3f}s" if tpe else "—"
            lines.append(
                f"| {short_label} "
                f"| {seq.elapsed:.3f}s "
                f"| {tpe_col} "
                f"| {pa_best.elapsed:.3f}s "
                f"| **{speedup:.1f}×** |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def _export_json(results: list[BenchmarkResult], path: Path):
    data = [
        {
            "category": r.category,
            "workload": r.workload,
            "runner": r.runner,
            "tasks": r.tasks,
            "elapsed_s": round(r.elapsed, 6),
            "throughput_tasks_per_s": round(r.throughput, 2),
        }
        for r in results
    ]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PyAccelerate Benchmark Suite")
    parser.add_argument("--io", action="store_true", help="IO-bound only")
    parser.add_argument("--cpu", action="store_true", help="CPU-bound only")
    parser.add_argument("--mixed", action="store_true", help="Mixed only")
    parser.add_argument("--quick", action="store_true", help="Fewer tasks (CI)")
    parser.add_argument("--repeats", type=int, default=3, help="Repetitions per bench")
    parser.add_argument("--json", type=str, default="", help="Export JSON path")
    args = parser.parse_args()

    run_all = not (args.io or args.cpu or args.mixed)

    workers = os.cpu_count() or 4

    if args.quick:
        n_io, n_cpu, n_compress, n_mixed = 200, 200, 100, 200
        n_io_var = 200
    else:
        n_io, n_cpu, n_compress, n_mixed = 1000, 500, 200, 500
        n_io_var = 1000

    all_results: list[BenchmarkResult] = []

    print("=" * 65)
    print("  PyAccelerate Benchmark Suite")
    print(f"  Python {sys.version.split()[0]} | CPUs: {os.cpu_count()} | Workers: {workers}")
    print("=" * 65)

    if run_all or args.io:
        print("\n▸ IO-bound benchmark …")
        all_results.extend(bench_io(n_io, args.repeats, workers))
        print("\n▸ IO-bound (variable latency) benchmark …")
        all_results.extend(bench_io_variable(n_io_var, args.repeats, workers))

    if run_all or args.cpu:
        print("\n▸ CPU-bound (SHA-256 chain) benchmark …")
        all_results.extend(bench_cpu_hash(n_cpu, args.repeats, workers))
        print("\n▸ CPU-bound (zlib compress) benchmark …")
        all_results.extend(bench_cpu_compress(n_compress, args.repeats, workers))

    if run_all or args.mixed:
        print("\n▸ Mixed workload benchmark …")
        all_results.extend(bench_mixed(n_mixed, args.repeats, workers))

    # Print detailed tables
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(_format_table(all_results))

    # JSON export
    out_dir = Path(__file__).resolve().parent
    json_path = Path(args.json) if args.json else out_dir / "results.json"
    _export_json(all_results, json_path)
    print(f"\n✓ Results exported to {json_path}")

    # Shutdown singletons
    from pyaccelerate.work_stealing import shutdown_scheduler
    from pyaccelerate.threads import shutdown_pools
    shutdown_scheduler(wait=True)
    shutdown_pools()


if __name__ == "__main__":
    main()
