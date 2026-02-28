#!/usr/bin/env python3
"""
Example 2: Parallel I/O with Public Data
=========================================

Downloads public datasets in parallel, processes them, and writes results.

Uses the Iris and Wine datasets from UCI ML Repository (via URLs).

Demonstrates:
  - Parallel HTTP downloads (I/O-bound)
  - Parallel CSV parsing / transformation
  - Parallel file writing
  - Profiling with @timed and Tracker
"""

import csv
import hashlib
import io
import json
import os
import tempfile
import time
import urllib.request
from typing import Any, Dict, List, Tuple

from pyaccelerate import Engine
from pyaccelerate.profiler import timed, Tracker, Timer
from pyaccelerate.threads import run_parallel, submit, batch_execute


# ── Public dataset URLs (small, always available) ──
DATASETS: Dict[str, str] = {
    "iris": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    "seeds": "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
    "glass": "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
    "ecoli": "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
}

# Fallback: generate synthetic data if downloads fail
FALLBACK_ROWS = 150


def download_dataset(name: str, url: str) -> Tuple[str, str, float]:
    """Download a dataset and return (name, content, elapsed_ms)."""
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pyaccelerate/0.3"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        elapsed = (time.perf_counter() - t0) * 1000
        return name, content, elapsed
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        # Generate synthetic fallback data
        rows = []
        for i in range(FALLBACK_ROWS):
            rows.append(",".join(str(round(i * 0.1 + j * 0.3, 2)) for j in range(5)))
        content = "\n".join(rows)
        return name, content, elapsed


def process_dataset(name: str, raw: str) -> Dict[str, Any]:
    """Parse CSV content and compute basic statistics."""
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    num_rows = len(lines)
    num_cols = 0
    numeric_values: List[float] = []

    for line in lines:
        parts = line.replace("\t", ",").split(",")
        num_cols = max(num_cols, len(parts))
        for p in parts:
            p = p.strip()
            try:
                numeric_values.append(float(p))
            except ValueError:
                pass

    md5 = hashlib.md5(raw.encode()).hexdigest()

    stats: Dict[str, Any] = {
        "name": name,
        "rows": num_rows,
        "columns": num_cols,
        "numeric_values": len(numeric_values),
        "md5": md5,
    }

    if numeric_values:
        stats["mean"] = round(sum(numeric_values) / len(numeric_values), 4)
        stats["min"] = round(min(numeric_values), 4)
        stats["max"] = round(max(numeric_values), 4)

    return stats


def save_result(output_dir: str, name: str, stats: Dict[str, Any]) -> str:
    """Save processed stats as JSON file."""
    path = os.path.join(output_dir, f"{name}_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return path


def main() -> None:
    engine = Engine()
    print(engine.status_line())
    print()

    output_dir = tempfile.mkdtemp(prefix="pyacc_example_")
    print(f"Output directory: {output_dir}\n")

    tracker = Tracker("parallel_io")

    # ── Stage 1: Parallel downloads ──
    print("=" * 60)
    print("  Stage 1: Parallel Downloads")
    print("=" * 60)

    download_items = [(name, url) for name, url in DATASETS.items()]

    with Timer("downloads") as t:
        downloaded = batch_execute(
            download_dataset,
            download_items,
            max_concurrent=len(DATASETS),  # all at once — I/O-bound
            desc="Downloading",
            show_progress=False,
        )

    for name, content, ms in downloaded:
        print(f"  {name:10s}: {len(content):>8,} bytes ({ms:.0f} ms)")
    print(f"  Total download time: {t.elapsed:.3f} s\n")

    # ── Stage 2: Parallel processing ──
    print("=" * 60)
    print("  Stage 2: Parallel Processing")
    print("=" * 60)

    process_items = [(name, content) for name, content, _ in downloaded]

    with tracker.measure():
        with Timer("processing") as t:
            stats_list = batch_execute(
                process_dataset,
                process_items,
                max_concurrent=engine.cpu_workers,
                desc="Processing",
                show_progress=False,
            )

    for stats in stats_list:
        print(f"  {stats['name']:10s}: {stats['rows']} rows, "
              f"{stats['numeric_values']} numeric values, "
              f"mean={stats.get('mean', 'N/A')}")
    print(f"  Total processing time: {t.elapsed:.3f} s\n")

    # ── Stage 3: Parallel writing ──
    print("=" * 60)
    print("  Stage 3: Parallel File Writing")
    print("=" * 60)

    write_items = [(output_dir, stats["name"], stats) for stats in stats_list]

    with Timer("writing") as t:
        paths = batch_execute(
            save_result,
            write_items,
            max_concurrent=len(write_items),
            desc="Writing",
            show_progress=False,
        )

    for path in paths:
        size = os.path.getsize(path)
        print(f"  {os.path.basename(path)}: {size} bytes")
    print(f"  Total write time: {t.elapsed:.3f} s\n")

    print(f"Tracker stats: {tracker.count} operations, "
          f"total={tracker.total:.3f} s, mean={tracker.mean:.3f} s")

    engine.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
