#!/usr/bin/env python3
"""
Example 6: Task Priority & Energy Management
==============================================

Demonstrates OS-level task priority and energy profile controls:
  - Setting process priority (IDLE → REALTIME)
  - Energy profile management (POWER_SAVER → ULTRA_PERFORMANCE)
  - Convenience presets
  - Priority info introspection
  - Impact on task execution times

Platform-aware: works on Windows, Linux, and macOS with graceful fallbacks.
"""

import math
import time
from typing import Any, Dict

from pyaccelerate.priority import (
    TaskPriority,
    EnergyProfile,
    set_task_priority,
    get_task_priority,
    set_energy_profile,
    get_energy_profile,
    get_priority_info,
    max_performance,
    balanced,
    power_saver,
)
from pyaccelerate.profiler import Timer


# ── Workload ────────────────────────────────────────────────────────────

def cpu_bench(iterations: int = 500_000) -> float:
    """Simple CPU benchmark: trig + sqrt loop."""
    total = 0.0
    for i in range(1, iterations + 1):
        total += math.sqrt(i) * math.sin(i) * math.cos(i)
    return total


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Priority & Energy Management")
    print("=" * 60)

    # ── 1. Inspect current priority info ──
    print("\n[1] Current Priority Info:")
    info = get_priority_info()
    for k, v in info.items():
        print(f"    {k}: {v}")

    # ── 2. Demonstrate task priority levels ──
    print("\n[2] Task Priority Levels:")
    results: Dict[str, float] = {}

    for priority in [TaskPriority.NORMAL, TaskPriority.HIGH, TaskPriority.ABOVE_NORMAL]:
        set_task_priority(priority)
        current = get_task_priority()
        print(f"\n  Priority: {priority.name} (actual: {current.name})")

        with Timer(f"bench-{priority.name}") as t:
            cpu_bench()
        results[priority.name] = t.elapsed
        print(f"    CPU bench: {t.elapsed:.4f} s")

    # Restore to NORMAL
    set_task_priority(TaskPriority.NORMAL)

    if results:
        baseline = results.get("NORMAL", 1.0)
        print(f"\n  Speedup vs NORMAL:")
        for name, elapsed in results.items():
            ratio = baseline / elapsed if elapsed > 0 else 0
            print(f"    {name:>15}: {ratio:.3f}x")

    # ── 3. Energy profiles ──
    print("\n[3] Energy Profiles:")

    for profile in EnergyProfile:
        try:
            set_energy_profile(profile)
            current = get_energy_profile()
            print(f"  Set: {profile.name:>20} → Actual: {current.name}")
        except (PermissionError, RuntimeError) as e:
            print(f"  Set: {profile.name:>20} → Skipped ({e})")

    # Restore balanced
    try:
        set_energy_profile(EnergyProfile.BALANCED)
    except Exception:
        pass

    # ── 4. Convenience presets ──
    print("\n[4] Convenience Presets:")

    presets = [
        ("power_saver()", power_saver),
        ("balanced()", balanced),
        ("max_performance()", max_performance),
    ]

    for label, fn in presets:
        try:
            fn()
            info = get_priority_info()
            print(f"\n  {label}")
            print(f"    Priority: {info['priority']}")
            print(f"    Energy:   {info['energy']}")
        except Exception as e:
            print(f"\n  {label} → Error: {e}")

    # Restore balanced
    try:
        balanced()
    except Exception:
        pass

    # ── 5. Priority under load ──
    print("\n\n[5] Priority Impact Under Load:")
    print("  Running benchmark at IDLE vs HIGH priority...\n")

    priorities_to_test = [TaskPriority.IDLE, TaskPriority.NORMAL, TaskPriority.HIGH]
    bench_results = []

    for priority in priorities_to_test:
        set_task_priority(priority)
        times = []
        for _ in range(3):
            with Timer(f"run-{priority.name}") as t:
                cpu_bench(200_000)
            times.append(t.elapsed)
        avg = sum(times) / len(times)
        bench_results.append((priority.name, avg))
        print(f"  {priority.name:>12}: avg={avg:.4f} s  (runs: {[f'{t:.4f}' for t in times]})")

    # Restore
    set_task_priority(TaskPriority.NORMAL)

    if bench_results:
        baseline = bench_results[1][1]  # NORMAL
        print(f"\n  Relative to NORMAL:")
        for name, avg in bench_results:
            ratio = avg / baseline if baseline > 0 else 0
            print(f"    {name:>12}: {ratio:.3f}x")

    print(f"\n  Final state:")
    info = get_priority_info()
    for k, v in info.items():
        print(f"    {k}: {v}")

    print("\nDone!")


if __name__ == "__main__":
    main()
