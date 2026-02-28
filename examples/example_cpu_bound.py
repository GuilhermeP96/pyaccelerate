#!/usr/bin/env python3
"""
Example 3: CPU-Bound Parallel Computing
========================================

Demonstrates CPU-bound parallel processing:
  - Mathematical computation across all cores
  - Process pool for GIL-free parallelism
  - Memory-aware worker clamping
  - Comparison: sequential vs threaded vs multiprocess
"""

import hashlib
import math
import os
import time
from typing import Dict, List, Tuple

from pyaccelerate import Engine
from pyaccelerate.cpu import detect, recommend_workers
from pyaccelerate.threads import get_pool, get_process_pool, batch_execute
from pyaccelerate.memory import clamp_workers, get_pressure
from pyaccelerate.profiler import Timer, Tracker


def compute_heavy(n: int) -> Dict[str, float]:
    """CPU-intensive computation: primes + trigonometry + hashing."""
    # Prime counting via sieve-like approach
    count = 0
    for i in range(2, n):
        is_prime = True
        for j in range(2, int(math.sqrt(i)) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            count += 1

    # Trigonometry series
    trig_sum = sum(math.sin(i) * math.cos(i) for i in range(n))

    # Hashing workload
    data = str(n).encode() * 100
    hash_result = hashlib.sha256(data).hexdigest()

    return {
        "n": n,
        "primes_below_n": count,
        "trig_sum": round(trig_sum, 6),
        "hash": hash_result[:16],
    }


def compute_chunk(start: int, end: int) -> float:
    """Pure math computation for a range (GIL-friendly when in process pool)."""
    total = 0.0
    for i in range(start, end):
        total += math.sqrt(i) * math.sin(i) * math.log(i + 1)
    return total


def main() -> None:
    engine = Engine()
    cpu_info = detect()
    print(engine.status_line())
    print(f"\nCPU: {cpu_info.short_label()}")
    print(f"Recommended CPU workers: {recommend_workers(io_bound=False)}")
    print(f"Recommended I/O workers: {recommend_workers(io_bound=True)}")
    print(f"Memory pressure: {get_pressure().name}")
    print()

    N = 5000  # complexity per task
    TASKS = cpu_info.logical_cores * 2
    items = [(N + i * 100,) for i in range(TASKS)]

    # ── 1. Sequential baseline ──
    print("=" * 60)
    print("  1. Sequential Execution")
    print("=" * 60)
    with Timer("sequential") as t_seq:
        seq_results = [compute_heavy(item[0]) for item in items]
    print(f"  {TASKS} tasks in {t_seq.elapsed:.3f} s")
    print(f"  Throughput: {TASKS / t_seq.elapsed:.1f} tasks/s\n")

    # ── 2. Thread pool (limited by GIL for CPU-bound) ──
    print("=" * 60)
    print("  2. Thread Pool (I/O pool)")
    print("=" * 60)
    with Timer("threaded") as t_thr:
        thr_results = batch_execute(
            compute_heavy,
            items,
            max_concurrent=cpu_info.logical_cores,
            desc="Threaded",
            show_progress=False,
        )
    print(f"  {TASKS} tasks in {t_thr.elapsed:.3f} s")
    print(f"  Throughput: {TASKS / t_thr.elapsed:.1f} tasks/s")
    speedup = t_seq.elapsed / t_thr.elapsed if t_thr.elapsed > 0 else 0
    print(f"  Speedup vs sequential: {speedup:.2f}x\n")

    # ── 3. Process pool (true parallelism, bypasses GIL) ──
    print("=" * 60)
    print("  3. Process Pool (CPU pool)")
    print("=" * 60)
    workers = clamp_workers(recommend_workers(io_bound=False))
    TOTAL_ITER = 2_000_000
    chunk_size = TOTAL_ITER // workers
    proc_items = [(i * chunk_size, (i + 1) * chunk_size) for i in range(workers)]

    with Timer("sequential-math") as t_seq_math:
        _ = compute_chunk(0, TOTAL_ITER)

    with Timer("process-pool") as t_proc:
        pool = get_process_pool()
        futures = [pool.submit(compute_chunk, s, e) for s, e in proc_items]
        proc_results = [f.result() for f in futures]

    print(f"  Sequential math: {t_seq_math.elapsed:.3f} s")
    print(f"  Process pool ({workers} workers): {t_proc.elapsed:.3f} s")
    speedup_proc = t_seq_math.elapsed / t_proc.elapsed if t_proc.elapsed > 0 else 0
    print(f"  Speedup: {speedup_proc:.2f}x\n")

    # ── Summary ──
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Sequential:   {t_seq.elapsed:.3f} s")
    print(f"  Threaded:     {t_thr.elapsed:.3f} s  ({TASKS / t_thr.elapsed:.1f} tasks/s)")
    print(f"  Process pool: {t_proc.elapsed:.3f} s  ({speedup_proc:.2f}x speedup)")

    engine.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
