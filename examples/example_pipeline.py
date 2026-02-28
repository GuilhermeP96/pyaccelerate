#!/usr/bin/env python3
"""
Example 5: Multi-stage Pipeline
================================

Shows how to build complex data pipelines combining:
  - Parallel downloads from public APIs
  - CPU-heavy data transformations
  - I/O-bound persistence
  - Profiling each stage independently

Uses public datasets and APIs for realistic workloads.
"""

import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
import time
import urllib.request
from typing import Any, Dict, List, Tuple

from pyaccelerate import Engine
from pyaccelerate.profiler import Timer, Tracker


# ── Stage 1: Download functions ─────────────────────────────────────────

DATASETS = {
    "iris": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    "wine": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    "seeds": "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
    "glass": "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
    "ecoli": "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data",
}


def download_dataset(name: str, url: str) -> Tuple[str, str]:
    """Download a dataset and return (name, content)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pyaccelerate/0.3"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        return name, text
    except Exception as e:
        # Generate synthetic fallback
        import random
        random.seed(hash(name))
        rows = "\n".join(
            ",".join(f"{random.uniform(0, 10):.2f}" for _ in range(5))
            for _ in range(150)
        )
        return name, rows


# ── Stage 2: Transform functions ────────────────────────────────────────

def parse_and_analyze(name: str, raw_text: str) -> Dict[str, Any]:
    """Parse CSV-like text, compute statistics per numeric column."""
    lines = [l.strip() for l in raw_text.strip().splitlines() if l.strip()]
    
    # Try to parse as CSV
    columns: List[List[float]] = []
    n_rows = 0
    for line in lines:
        parts = line.replace("\t", ",").split(",")
        nums: List[float] = []
        for p in parts:
            p = p.strip()
            try:
                nums.append(float(p))
            except ValueError:
                continue
        if nums:
            while len(columns) < len(nums):
                columns.append([])
            for i, v in enumerate(nums):
                columns[i].append(v)
            n_rows += 1

    stats = {}
    for i, col in enumerate(columns):
        if len(col) >= 2:
            stats[f"col_{i}"] = {
                "mean": round(statistics.mean(col), 4),
                "stdev": round(statistics.stdev(col), 4),
                "min": min(col),
                "max": max(col),
                "median": round(statistics.median(col), 4),
                "count": len(col),
            }

    # Simulate CPU-heavy additional computation
    checksum = 0.0
    for col in columns:
        for v in col:
            checksum += math.log1p(abs(v)) * math.cos(v)

    return {
        "name": name,
        "rows": n_rows,
        "numeric_columns": len(columns),
        "column_stats": stats,
        "checksum": round(checksum, 6),
        "md5": hashlib.md5(raw_text.encode()).hexdigest(),
    }


# ── Stage 3: Persistence functions ─────────────────────────────────────

def write_report(directory: str, analysis: Dict[str, Any]) -> str:
    """Write analysis as JSON report."""
    name = analysis["name"]
    path = os.path.join(directory, f"{name}_report.json")
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
    return path


def write_csv_summary(directory: str, analyses: List[Dict]) -> str:
    """Write combined summary CSV."""
    path = os.path.join(directory, "pipeline_summary.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "rows", "columns", "checksum", "md5"])
        for a in analyses:
            writer.writerow([
                a["name"], a["rows"], a["numeric_columns"],
                a["checksum"], a["md5"],
            ])
    return path


# ── Main ────────────────────────────────────────────────────────────────

def main() -> None:
    output_dir = tempfile.mkdtemp(prefix="pyacc_pipeline_")
    tracker = Tracker()

    print("=" * 60)
    print("  Pipeline Example: Download → Analyze → Report")
    print("=" * 60)

    engine = Engine()
    print(f"\n{engine.summary()}\n")

    # ── Stage 1: Parallel download ──
    print("Stage 1: Downloading datasets...")
    with Timer("download") as t_dl:
        downloads = engine.batch(
            download_dataset,
            [(name, url) for name, url in DATASETS.items()],
            desc="Download",
            show_progress=False,
        )
    tracker.record("download", t_dl.elapsed, len(downloads))
    for name, text in downloads:
        print(f"  {name:>8}: {len(text):>8,} chars")

    # ── Stage 2: Parallel analysis ──
    print("\nStage 2: Analyzing datasets...")
    with Timer("analyze") batch(
            parse_and_analyze,
            [(name, text) for name, text in downloads],
            desc="Analyze",
            show_progress=False,   for name, text in downloads
            ]
        )
    tracker.record("analyze", t_an.elapsed, len(analyses))
    for a in analyses:
        cols = a["numeric_columns"]
        print(f"  {a['name']:>8}: {a['rows']:>4} rows × {cols} cols  (checksum={a['checksum']:.3f})")

    # ── Stage 3: Parallel persistence ──
    print("\nStage 3: Writing reports...")
    with Timer("persistbatch(
            write_report,
            [(output_dir, a) for a in analyses],
            desc="Write",
            show_progress=False,
            [(write_report, output_dir, a) for a in analyses]
        )
        summary_path = write_csv_summary(output_dir, analyses)
    tracker.record("persist", t_wr.elapsed, len(paths) + 1)
    for p in paths:
        print(f"  Written: {os.path.basename(p)}")
    print(f"  Summary: {os.path.basename(summary_path)}")

    # ── Results ──
    print(f"\n{'─' * 60}")
    total = t_dl.elapsed + t_an.elapsed + t_wr.elapsed
    print(f"  Download : {t_dl.elapsed:>8.3f} s")
    print(f"  Analyze  : {t_an.elapsed:>8.3f} s")
    print(f"  Persist  : {t_wr.elapsed:>8.3f} s")
    print(f"  TOTAL    : {total:>8.3f} s")
    print(f"\n  Output: {output_dir}")

    engine.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
