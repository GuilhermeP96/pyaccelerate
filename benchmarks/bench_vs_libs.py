"""
PyAccelerate vs Popular Libraries — Head-to-Head Benchmark
============================================================

Compares **separate, specialised libraries** (each best-in-class for its
domain) against **pyaccelerate as a single unified runtime**.

Libraries under test
--------------------
  IO-bound:   asyncio, ThreadPoolExecutor, joblib(threads)
  CPU-bound:  multiprocessing, joblib(loky), ray
  Mixed:      manual asyncio+ProcessPool combo

pyaccelerate runners
--------------------
  threads, work-stealing, engine, adaptive

The key insight:  other projects force you to *pick the right tool* for each
workload.  pyaccelerate gives you **one API** that adapts automatically.

Usage:
    python -m benchmarks.bench_vs_libs              # full
    python -m benchmarks.bench_vs_libs --quick       # CI / fast
    python -m benchmarks.bench_vs_libs --io          # IO only
    python -m benchmarks.bench_vs_libs --cpu         # CPU only
    python -m benchmarks.bench_vs_libs --mixed       # mixed only
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ---------------------------------------------------------------------------
# local src/ on path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import pyaccelerate as pa                        # noqa: E402
from pyaccelerate.threads import map_parallel    # noqa: E402
from pyaccelerate.work_stealing import ws_map    # noqa: E402

# ---------------------------------------------------------------------------
# optional imports — gracefully degrade
# ---------------------------------------------------------------------------
try:
    import joblib                                 # noqa: E402
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

try:
    import ray                                    # noqa: E402
    _HAS_RAY = True
except ImportError:
    _HAS_RAY = False

# ---------------------------------------------------------------------------
# Workload functions (module-level for pickling)
# ---------------------------------------------------------------------------
_IO_SLEEP = 0.02
_HASH_ROUNDS = 800
_COMPRESS_SIZE = 200_000


def io_task(idx: int) -> int:
    time.sleep(_IO_SLEEP)
    return idx


async def _aio_task(idx: int) -> int:
    await asyncio.sleep(_IO_SLEEP)
    return idx


def cpu_hash_task(idx: int) -> bytes:
    data = struct.pack(">I", idx)
    for _ in range(_HASH_ROUNDS):
        data = hashlib.sha256(data).digest()
    return data


def cpu_compress_task(idx: int) -> int:
    blob = bytes(range(256)) * (_COMPRESS_SIZE // 256) + struct.pack(">I", idx)
    return len(zlib.compress(blob, 6))


def mixed_task(idx: int) -> int:
    time.sleep(_IO_SLEEP / 2)
    data = struct.pack(">I", idx)
    for _ in range(_HASH_ROUNDS // 2):
        data = hashlib.sha256(data).digest()
    return idx


# ---------------------------------------------------------------------------
# Runner wrappers
# ---------------------------------------------------------------------------

def _sequential(fn, items):
    return [fn(i) for i in items]


# --- stdlib ---
def _threadpool(fn, items, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fn, i) for i in items]
        return [f.result() for f in futs]


# --- asyncio ---
def _asyncio_gather(items):
    async def _go():
        return await asyncio.gather(*[_aio_task(i) for i in items])
    return asyncio.run(_go())


# --- joblib ---
def _joblib_threads(fn, items, workers):
    return joblib.Parallel(n_jobs=workers, backend="threading")(
        joblib.delayed(fn)(i) for i in items
    )


# --- ray ---
def _ray_remote(fn, items):
    remote_fn = ray.remote(fn)
    refs = [remote_fn.remote(i) for i in items]
    return ray.get(refs)


# --- pyaccelerate ---
def _pa_threads(fn, items):
    return map_parallel(fn, [(i,) for i in items])


def _pa_ws(fn, items):
    return ws_map(fn, [(i,) for i in items])


def _pa_engine(fn, items):
    engine = pa.Engine(auto_threads=True)
    try:
        return engine.batch(fn, [(i,) for i in items], show_progress=False)
    finally:
        engine.shutdown()


def _pa_adaptive(fn, items):
    with pa.AdaptiveScheduler() as sched:
        return sched.map(fn, [(i,) for i in items])


# ---------------------------------------------------------------------------
# Timing + result container
# ---------------------------------------------------------------------------

class Result:
    __slots__ = ("category", "workload", "runner", "lib_type", "tasks",
                 "elapsed", "throughput")

    def __init__(self, category: str, workload: str, runner: str,
                 lib_type: str, tasks: int, elapsed: float):
        self.category = category
        self.workload = workload
        self.runner = runner
        self.lib_type = lib_type
        self.tasks = tasks
        self.elapsed = elapsed
        self.throughput = tasks / elapsed if elapsed > 0 else float("inf")


def _bench(category, workload, runner, lib_type, fn_call, tasks, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn_call()
        times.append(time.perf_counter() - t0)
    best = min(times)
    return Result(category, workload, runner, lib_type, tasks, best)


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

def suite_io(n: int, reps: int, workers: int) -> list[Result]:
    items = list(range(n))
    results = []
    cat = "IO-bound"
    wl = f"sleep {_IO_SLEEP*1000:.0f}ms x {n}"

    results.append(_bench(cat, wl, "Sequential", "baseline",
                          lambda: _sequential(io_task, items), n, reps))
    results.append(_bench(cat, wl, "ThreadPoolExecutor", "stdlib",
                          lambda: _threadpool(io_task, items, workers), n, reps))
    results.append(_bench(cat, wl, "asyncio.gather", "stdlib",
                          lambda: _asyncio_gather(items), n, reps))
    if _HAS_JOBLIB:
        results.append(_bench(cat, wl, "joblib(threads)", "3rd-party",
                              lambda: _joblib_threads(io_task, items, workers), n, reps))
    results.append(_bench(cat, wl, "pyaccelerate.threads", "pyaccelerate",
                          lambda: _pa_threads(io_task, items), n, reps))
    results.append(_bench(cat, wl, "pyaccelerate.ws", "pyaccelerate",
                          lambda: _pa_ws(io_task, items), n, reps))
    results.append(_bench(cat, wl, "pyaccelerate.engine", "pyaccelerate",
                          lambda: _pa_engine(io_task, items), n, reps))
    return results


def suite_cpu(n: int, reps: int, workers: int) -> list[Result]:
    items = list(range(n))
    results = []
    cat = "CPU-bound"
    wl = f"zlib {_COMPRESS_SIZE//1000}KB x {n}"

    results.append(_bench(cat, wl, "Sequential", "baseline",
                          lambda: _sequential(cpu_compress_task, items), n, reps))
    results.append(_bench(cat, wl, "ThreadPoolExecutor", "stdlib",
                          lambda: _threadpool(cpu_compress_task, items, workers), n, reps))
    if _HAS_JOBLIB:
        results.append(_bench(cat, wl, "joblib(threads)", "3rd-party",
                              lambda: _joblib_threads(cpu_compress_task, items, workers), n, reps))
    if _HAS_RAY:
        results.append(_bench(cat, wl, "ray.remote", "3rd-party",
                              lambda: _ray_remote(cpu_compress_task, items), n, reps))
    results.append(_bench(cat, wl, "pyaccelerate.threads", "pyaccelerate",
                          lambda: _pa_threads(cpu_compress_task, items), n, reps))
    results.append(_bench(cat, wl, "pyaccelerate.ws", "pyaccelerate",
                          lambda: _pa_ws(cpu_compress_task, items), n, reps))
    results.append(_bench(cat, wl, "pyaccelerate.engine", "pyaccelerate",
                          lambda: _pa_engine(cpu_compress_task, items), n, reps))
    return results


def suite_cpu_hash(n: int, reps: int, workers: int) -> list[Result]:
    items = list(range(n))
    results = []
    cat = "CPU-bound"
    wl = f"SHA-256 x{_HASH_ROUNDS} x {n}"

    results.append(_bench(cat, wl, "Sequential", "baseline",
                          lambda: _sequential(cpu_hash_task, items), n, reps))
    results.append(_bench(cat, wl, "ThreadPoolExecutor", "stdlib",
                          lambda: _threadpool(cpu_hash_task, items, workers), n, reps))
    if _HAS_JOBLIB:
        results.append(_bench(cat, wl, "joblib(threads)", "3rd-party",
                              lambda: _joblib_threads(cpu_hash_task, items, workers), n, reps))
    if _HAS_RAY:
        results.append(_bench(cat, wl, "ray.remote", "3rd-party",
                              lambda: _ray_remote(cpu_hash_task, items), n, reps))
    results.append(_bench(cat, wl, "pyaccelerate.threads", "pyaccelerate",
                          lambda: _pa_threads(cpu_hash_task, items), n, reps))
    results.append(_bench(cat, wl, "pyaccelerate.ws", "pyaccelerate",
                          lambda: _pa_ws(cpu_hash_task, items), n, reps))
    return results


def suite_mixed(n: int, reps: int, workers: int) -> list[Result]:
    items = list(range(n))
    results = []
    cat = "Mixed"
    wl = f"IO({_IO_SLEEP*500:.0f}ms)+CPU(SHA x{_HASH_ROUNDS//2}) x {n}"

    results.append(_bench(cat, wl, "Sequential", "baseline",
                          lambda: _sequential(mixed_task, items), n, reps))
    results.append(_bench(cat, wl, "ThreadPoolExecutor", "stdlib",
                          lambda: _threadpool(mixed_task, items, workers), n, reps))
    if _HAS_JOBLIB:
        results.append(_bench(cat, wl, "joblib(threads)", "3rd-party",
                              lambda: _joblib_threads(mixed_task, items, workers), n, reps))
    if _HAS_RAY:
        results.append(_bench(cat, wl, "ray.remote", "3rd-party",
                              lambda: _ray_remote(mixed_task, items), n, reps))
    results.append(_bench(cat, wl, "pyaccelerate.threads", "pyaccelerate",
                          lambda: _pa_threads(mixed_task, items), n, reps))
    results.append(_bench(cat, wl, "pyaccelerate.ws", "pyaccelerate",
                          lambda: _pa_ws(mixed_task, items), n, reps))
    results.append(_bench(cat, wl, "pyaccelerate.engine", "pyaccelerate",
                          lambda: _pa_engine(mixed_task, items), n, reps))
    results.append(_bench(cat, wl, "pyaccelerate.adaptive", "pyaccelerate",
                          lambda: _pa_adaptive(mixed_task, items), n, reps))
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format(results: list[Result]) -> str:
    groups: dict[str, list[Result]] = {}
    for r in results:
        key = f"{r.category}: {r.workload}"
        groups.setdefault(key, []).append(r)

    lines: list[str] = []
    for label, rows in groups.items():
        rows.sort(key=lambda r: r.elapsed)
        seq_time = next((r.elapsed for r in rows if r.runner == "Sequential"),
                        rows[-1].elapsed)

        lines.append(f"\n### {label}\n")
        lines.append("| # | Runner | Type | Time (s) | Speedup | Tasks/sec |")
        lines.append("|---|--------|------|----------|---------|-----------|")
        for i, r in enumerate(rows, 1):
            speedup = seq_time / r.elapsed if r.elapsed > 0 else float("inf")
            icon = {"baseline": " ", "stdlib": "S", "3rd-party": "L",
                    "pyaccelerate": "P"}.get(r.lib_type, " ")
            lines.append(
                f"| {i} | {r.runner:<24s} | {icon} "
                f"| {r.elapsed:>8.3f} | {speedup:>6.1f}x "
                f"| {r.throughput:>9.0f} |"
            )

        # summary line
        best_ext = min((r for r in rows if r.lib_type in ("stdlib", "3rd-party")),
                       key=lambda r: r.elapsed, default=None)
        best_pa = min((r for r in rows if r.lib_type == "pyaccelerate"),
                      key=lambda r: r.elapsed, default=None)
        if best_ext and best_pa:
            diff_pct = (best_ext.elapsed - best_pa.elapsed) / best_ext.elapsed * 100
            if diff_pct > 0:
                lines.append(f"\n> **pyaccelerate ({best_pa.runner})** is "
                             f"**{diff_pct:.0f}% faster** than best external "
                             f"({best_ext.runner})")
            else:
                lines.append(f"\n> Best external ({best_ext.runner}) is "
                             f"{-diff_pct:.0f}% faster than pyaccelerate "
                             f"({best_pa.runner})")

    lines.append("\n\n*Type: S=stdlib, L=3rd-party lib, P=pyaccelerate*")
    return "\n".join(lines)


def _export_json(results: list[Result], path: Path):
    data = [
        {
            "category": r.category,
            "workload": r.workload,
            "runner": r.runner,
            "lib_type": r.lib_type,
            "tasks": r.tasks,
            "elapsed_s": round(r.elapsed, 6),
            "throughput": round(r.throughput, 2),
        }
        for r in results
    ]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PyAccelerate vs Popular Libraries Benchmark")
    parser.add_argument("--io", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    run_all = not (args.io or args.cpu or args.mixed)
    workers = os.cpu_count() or 4

    if args.quick:
        n_io, n_cpu, n_hash, n_mixed = 200, 100, 200, 200
    else:
        n_io, n_cpu, n_hash, n_mixed = 500, 200, 500, 300

    # Ray init (once)
    if _HAS_RAY:
        ray.init(ignore_reinit_error=True, logging_level="ERROR",
                 num_cpus=workers)

    all_results: list[Result] = []

    print("=" * 70)
    print("  PyAccelerate vs Popular Libraries")
    print(f"  Python {sys.version.split()[0]} | CPUs: {os.cpu_count()}")
    avail = []
    if _HAS_JOBLIB:
        avail.append(f"joblib {joblib.__version__}")
    if _HAS_RAY:
        avail.append(f"ray {ray.__version__}")
    print(f"  Libs: {', '.join(avail) if avail else 'none (install joblib/ray)'}")
    print("=" * 70)

    if run_all or args.io:
        print("\n>>> IO-bound ...")
        all_results.extend(suite_io(n_io, args.repeats, workers))

    if run_all or args.cpu:
        print("\n>>> CPU-bound (zlib) ...")
        all_results.extend(suite_cpu(n_cpu, args.repeats, workers))
        print("\n>>> CPU-bound (SHA-256 hash chain) ...")
        all_results.extend(suite_cpu_hash(n_hash, args.repeats, workers))

    if run_all or args.mixed:
        print("\n>>> Mixed IO+CPU ...")
        all_results.extend(suite_mixed(n_mixed, args.repeats, workers))

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(_format(all_results))

    out_dir = Path(__file__).resolve().parent
    json_path = Path(args.json) if args.json else out_dir / "vs_libs_results.json"
    _export_json(all_results, json_path)
    print(f"\n=> Results: {json_path}")

    # cleanup
    from pyaccelerate.work_stealing import shutdown_scheduler
    from pyaccelerate.threads import shutdown_pools
    shutdown_scheduler(wait=True)
    shutdown_pools()
    if _HAS_RAY:
        ray.shutdown()


if __name__ == "__main__":
    main()
