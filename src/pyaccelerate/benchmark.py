"""
pyaccelerate.benchmark — Built-in micro-benchmarks for the current host.

Runs quick benchmarks to characterise host performance:
  - CPU single-thread & multi-thread throughput
  - Memory bandwidth
  - I/O thread pool latency
  - GPU compute throughput (if available)

Results are returned as structured dicts suitable for logging, dashboards
or automated tuning decisions.

Usage::

    from pyaccelerate.benchmark import run_all, run_cpu, run_gpu

    report = run_all()
    print(report)
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

log = logging.getLogger("pyaccelerate.benchmark")


# ═══════════════════════════════════════════════════════════════════════════
#  CPU benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def run_cpu(
    iterations: int = 500_000,
    hash_bytes: int = 4096,
) -> Dict[str, Any]:
    """Single-thread CPU throughput:  math + hashing workload."""
    # Math workload
    t0 = time.perf_counter()
    total = 0.0
    for i in range(1, iterations + 1):
        total += math.sqrt(i) * math.sin(i)
    math_time = time.perf_counter() - t0

    # Hash workload
    data = os.urandom(hash_bytes)
    t0 = time.perf_counter()
    for _ in range(iterations):
        hashlib.md5(data).hexdigest()
    hash_time = time.perf_counter() - t0

    return {
        "benchmark": "cpu_single_thread",
        "iterations": iterations,
        "math_time_s": round(math_time, 4),
        "hash_time_s": round(hash_time, 4),
        "total_s": round(math_time + hash_time, 4),
        "math_ops_per_sec": int(iterations / math_time) if math_time > 0 else 0,
        "hash_ops_per_sec": int(iterations / hash_time) if hash_time > 0 else 0,
    }


def run_cpu_multithread(
    iterations: int = 200_000,
    workers: int = 0,
) -> Dict[str, Any]:
    """Multi-thread CPU throughput using standard thread pool."""
    if workers <= 0:
        workers = os.cpu_count() or 4

    def _work(start: int, end: int) -> float:
        total = 0.0
        for i in range(start, end):
            total += math.sqrt(i) * math.sin(i)
        return total

    chunk = iterations // workers
    ranges = [(i * chunk, (i + 1) * chunk) for i in range(workers)]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_work, s, e) for s, e in ranges]
        results = [f.result() for f in futures]
    elapsed = time.perf_counter() - t0

    return {
        "benchmark": "cpu_multi_thread",
        "iterations": iterations,
        "workers": workers,
        "time_s": round(elapsed, 4),
        "ops_per_sec": int(iterations / elapsed) if elapsed > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Thread pool latency
# ═══════════════════════════════════════════════════════════════════════════

def run_thread_pool_latency(
    tasks: int = 1000,
    pool_size: int = 0,
) -> Dict[str, Any]:
    """Measure submit→complete latency for no-op tasks on the I/O pool."""
    from pyaccelerate.threads import get_pool, io_pool_size

    if pool_size <= 0:
        pool_size = io_pool_size()

    pool = get_pool()
    latencies: list[float] = []

    for _ in range(tasks):
        t0 = time.perf_counter()
        fut = pool.submit(lambda: None)
        fut.result()
        latencies.append(time.perf_counter() - t0)

    avg = sum(latencies) / len(latencies) if latencies else 0
    p95_idx = int(len(latencies) * 0.95)
    sorted_lat = sorted(latencies)

    return {
        "benchmark": "thread_pool_latency",
        "tasks": tasks,
        "pool_size": pool_size,
        "avg_latency_us": round(avg * 1_000_000, 1),
        "p95_latency_us": round(sorted_lat[p95_idx] * 1_000_000, 1) if sorted_lat else 0,
        "min_latency_us": round(sorted_lat[0] * 1_000_000, 1) if sorted_lat else 0,
        "max_latency_us": round(sorted_lat[-1] * 1_000_000, 1) if sorted_lat else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Memory bandwidth
# ═══════════════════════════════════════════════════════════════════════════

def run_memory_bandwidth(
    size_mb: int = 64,
    iterations: int = 10,
) -> Dict[str, Any]:
    """Measure sequential memory read/write bandwidth."""
    size = size_mb * 1024 * 1024

    # Write
    t0 = time.perf_counter()
    for _ in range(iterations):
        buf = bytearray(size)
    write_time = time.perf_counter() - t0

    # Read (hash to prevent optimization)
    t0 = time.perf_counter()
    for _ in range(iterations):
        _h = hashlib.md5(buf).digest()
    read_time = time.perf_counter() - t0

    total_bytes = size * iterations
    return {
        "benchmark": "memory_bandwidth",
        "size_mb": size_mb,
        "iterations": iterations,
        "write_gbps": round(total_bytes / write_time / (1024 ** 3), 2) if write_time > 0 else 0,
        "read_gbps": round(total_bytes / read_time / (1024 ** 3), 2) if read_time > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  GPU benchmark
# ═══════════════════════════════════════════════════════════════════════════

def run_gpu(
    size: int = 10_000_000,
    iterations: int = 100,
) -> Dict[str, Any]:
    """GPU compute throughput: element-wise operations on a large array.

    Falls back to a CPU result if no GPU backend is available.
    """
    from pyaccelerate.gpu import gpu_available, best_gpu

    if not gpu_available():
        return {
            "benchmark": "gpu_compute",
            "available": False,
            "note": "No usable GPU — skipped",
        }

    gpu = best_gpu()
    backend = gpu.backend if gpu else "none"

    if backend == "cuda":
        return _bench_cuda(size, iterations, gpu)
    elif backend == "opencl":
        return _bench_opencl(size, iterations, gpu)
    elif backend == "intel":
        return _bench_intel(size, iterations, gpu)

    return {"benchmark": "gpu_compute", "available": False, "note": "Unsupported backend"}


def _bench_cuda(size: int, iterations: int, gpu: Any) -> Dict[str, Any]:
    try:
        import cupy as cp  # type: ignore[import-untyped]
        with cp.cuda.Device(gpu._index):
            a = cp.random.random(size, dtype=cp.float32)
            b = cp.random.random(size, dtype=cp.float32)
            cp.cuda.Device(gpu._index).synchronize()

            t0 = time.perf_counter()
            for _ in range(iterations):
                c = a * b + a
            cp.cuda.Device(gpu._index).synchronize()
            elapsed = time.perf_counter() - t0

        ops = size * 2 * iterations  # mul + add
        return {
            "benchmark": "gpu_compute",
            "available": True,
            "backend": "cuda",
            "device": gpu.name,
            "elements": size,
            "iterations": iterations,
            "time_s": round(elapsed, 4),
            "gflops": round(ops / elapsed / 1e9, 2) if elapsed > 0 else 0,
        }
    except Exception as exc:
        return {"benchmark": "gpu_compute", "available": True, "error": str(exc)}


def _bench_opencl(size: int, iterations: int, gpu: Any) -> Dict[str, Any]:
    try:
        import numpy as np  # type: ignore[import-untyped]
        a = np.random.random(size).astype(np.float32)
        b = np.random.random(size).astype(np.float32)

        t0 = time.perf_counter()
        for _ in range(iterations):
            c = a * b + a
        elapsed = time.perf_counter() - t0

        ops = size * 2 * iterations
        return {
            "benchmark": "gpu_compute",
            "available": True,
            "backend": "opencl",
            "device": gpu.name,
            "elements": size,
            "iterations": iterations,
            "time_s": round(elapsed, 4),
            "gflops": round(ops / elapsed / 1e9, 2) if elapsed > 0 else 0,
            "note": "OpenCL benchmark uses numpy host-side as proxy",
        }
    except Exception as exc:
        return {"benchmark": "gpu_compute", "available": True, "error": str(exc)}


def _bench_intel(size: int, iterations: int, gpu: Any) -> Dict[str, Any]:
    try:
        import dpnp  # type: ignore[import-untyped]
        a = dpnp.random.random(size).astype(dpnp.float32)
        b = dpnp.random.random(size).astype(dpnp.float32)

        t0 = time.perf_counter()
        for _ in range(iterations):
            c = a * b + a
        elapsed = time.perf_counter() - t0

        ops = size * 2 * iterations
        return {
            "benchmark": "gpu_compute",
            "available": True,
            "backend": "intel",
            "device": gpu.name,
            "elements": size,
            "iterations": iterations,
            "time_s": round(elapsed, 4),
            "gflops": round(ops / elapsed / 1e9, 2) if elapsed > 0 else 0,
        }
    except Exception as exc:
        return {"benchmark": "gpu_compute", "available": True, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
#  Full suite
# ═══════════════════════════════════════════════════════════════════════════

def run_all(quick: bool = True) -> Dict[str, Any]:
    """Run all benchmarks and return a combined report.

    Parameters
    ----------
    quick : bool
        If True, use reduced iteration counts for a faster run (~5 s).
    """
    scale = 1 if quick else 5

    results: Dict[str, Any] = {}
    results["cpu_single"] = run_cpu(iterations=100_000 * scale)
    results["cpu_multi"] = run_cpu_multithread(iterations=100_000 * scale)
    results["thread_latency"] = run_thread_pool_latency(tasks=200 * scale)
    results["memory"] = run_memory_bandwidth(size_mb=16 * scale, iterations=3 * scale)
    results["gpu"] = run_gpu(size=1_000_000 * scale, iterations=20 * scale)

    return results
