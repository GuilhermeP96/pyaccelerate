#!/usr/bin/env python3
"""
Example 4: Maximum Optimization Mode
======================================

Demonstrates the MaxMode that activates ALL resources simultaneously:
  - OS priority & energy optimization
  - Parallel CPU + I/O + GPU workloads
  - Pipeline execution
  - Real-time performance monitoring

Uses public data endpoints for I/O workloads.
"""

import hashlib
import json
import math
import os
import tempfile
import time
import urllib.request
from typing import Any, Dict, List, Tuple

from pyaccelerate.max_mode import MaxMode, activate_max_mode, deactivate_max_mode
from pyaccelerate.profiler import Timer, Tracker


# ── Workload functions ──────────────────────────────────────────────────

def cpu_work(n: int) -> float:
    """CPU-intensive: compute sum of sqrt * sin for range."""
    return sum(math.sqrt(i) * math.sin(i) for i in range(n))


def io_work(url: str) -> int:
    """I/O-intensive: download and hash content."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pyaccelerate/0.3"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        return len(data)
    except Exception:
        # Fallback: simulate I/O with sleep + hash
        time.sleep(0.05)
        return len(hashlib.sha256(url.encode()).digest())


def transform_data(text: str) -> Dict[str, Any]:
    """Transform raw text data into statistics."""
    words = text.split()
    lines = text.splitlines()
    return {
        "chars": len(text),
        "words": len(words),
        "lines": len(lines),
        "hash": hashlib.md5(text.encode()).hexdigest()[:12],
    }


def save_json(directory: str, name: str, data: Dict) -> str:
    """Save data as JSON file."""
    path = os.path.join(directory, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ── Public URLs for I/O tests ──────────────────────────────────────────

IO_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names",
    "https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
]


def main() -> None:
    output_dir = tempfile.mkdtemp(prefix="pyacc_maxmode_")

    # ═══════════════════════════════════════════════════════════════════
    #  Method 1: Context Manager (recommended)
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  MaxMode — Context Manager")
    print("=" * 60)

    with MaxMode(set_priority=True, set_energy=False) as m:
        print(m.summary())
        print()

        # ── Run all workloads simultaneously ──
        print("Running CPU + I/O workloads in parallel...")
        results = m.run_all(
            cpu_fn=cpu_work,
            cpu_items=[(50_000 + i * 10_000,) for i in range(8)],
            io_fn=io_work,
            io_items=[(url,) for url in IO_URLS],
        )

        print(f"\n  CPU results: {len(results.get('cpu', []))} tasks")
        print(f"  I/O results: {len(results.get('io', []))} tasks")
        print(f"  Total elapsed: {results['elapsed_s']:.3f} s")

        # ── Run I/O only ──
        print("\nRunning I/O-only download...")
        with Timer("io-only") as t:
            io_results = m.run_io(
                io_work,
                [(url,) for url in IO_URLS[:4]],
                desc="Downloads",
                show_progress=False,
            )
        print(f"  Downloaded {len(io_results)} files in {t.elapsed:.3f} s")
        print(f"  Total bytes: {sum(r for r in io_results if isinstance(r, int)):,}")

        # ── Run CPU only ──
        print("\nRunning CPU-only computation...")
        with Timer("cpu-only") as t:
            cpu_results = m.run_cpu(
                cpu_work,
                [(100_000,) for _ in range(8)],
            )
        print(f"  Computed {len(cpu_results)} tasks in {t.elapsed:.3f} s")

    print("\nMaxMode deactivated.\n")

    # ═══════════════════════════════════════════════════════════════════
    #  Method 2: Manual activation (for long-running services)
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("  MaxMode — Manual Activation")
    print("=" * 60)

    state = activate_max_mode(set_priority=True, set_energy=False)
    print(f"  Manifest: {json.dumps(state.hardware_manifest, indent=2)}")

    # Do work...
    with Timer("manual-work") as t:
        results = [cpu_work(50_000) for _ in range(4)]
    print(f"  Work completed in {t.elapsed:.3f} s")

    summary = deactivate_max_mode(state)
    print(f"  Deactivation summary: {summary}")

    # ═══════════════════════════════════════════════════════════════════
    #  Method 3: Pipeline execution
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  MaxMode — Pipeline (Download → Transform → Save)")
    print("=" * 60)

    def download_text(url: str) -> str:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pyaccelerate/0.3"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:
            return f"fallback data for {url}"

    with MaxMode(set_priority=True, set_energy=False) as m:
        # Download stage
        texts = m.run_io(
            download_text,
            [(url,) for url in IO_URLS[:4]],
            desc="Download",
            show_progress=False,
        )

        # Transform stage
        transformed = m.run_io(
            transform_data,
            [(t,) for t in texts],
            desc="Transform",
            show_progress=False,
        )

        # Save stage
        names = [f"dataset_{i}" for i in range(len(transformed))]
        saved = m.run_io(
            save_json,
            [(output_dir, n, d) for n, d in zip(names, transformed)],
            desc="Save",
            show_progress=False,
        )

        print(f"\n  Pipeline results:")
        for i, path in enumerate(saved):
            print(f"    [{i}] {os.path.basename(path)}: {transformed[i].get('chars', 0):,} chars")

    print(f"\n  Output: {output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
