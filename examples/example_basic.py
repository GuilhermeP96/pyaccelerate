#!/usr/bin/env python3
"""
Example 1: Basic Engine Usage
==============================

Demonstrates the fundamental pyaccelerate workflow:
  - Hardware detection
  - Engine summary
  - Single task submission
  - Simple parallel execution
"""

from pyaccelerate import Engine


def main() -> None:
    # ── 1. Create the engine (auto-detects all hardware) ──
    engine = Engine()

    # ── 2. Print full hardware report ──
    print(engine.summary())
    print()

    # ── 3. One-line status ──
    print("Status:", engine.status_line())
    print()

    # ── 4. Submit a single task ──
    future = engine.submit(lambda x: x ** 2, 42)
    print(f"Single task result: {future.result()}")

    # ── 5. Run parallel tasks ──
    items = [(i,) for i in range(20)]
    completed = engine.run_parallel(
        lambda x: x * 2,
        items,
        max_concurrent=8,
    )
    print(f"Parallel tasks completed: {completed}")

    # ── 6. Batch execute with progress ──
    results = engine.batch(
        lambda x: x ** 3,
        [(i,) for i in range(100)],
        max_concurrent=12,
        desc="Cubing numbers",
        show_progress=False,
    )
    print(f"Batch results (first 10): {results[:10]}")

    # ── 7. Machine-readable dict ──
    info = engine.as_dict()
    print(f"\nCPU cores: {info['cpu']['physical_cores']}P / {info['cpu']['logical_cores']}L")
    print(f"Memory pressure: {info['memory_pressure']}")
    print(f"GPU enabled: {info['gpu']['enabled']}")
    print(f"NPU enabled: {info['npu']['enabled']}")

    # ── 8. Shutdown ──
    engine.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
