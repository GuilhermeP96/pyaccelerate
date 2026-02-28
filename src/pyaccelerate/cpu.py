"""
pyaccelerate.cpu — CPU detection, topology and affinity utilities.

Provides runtime introspection of the host CPU capabilities:
  - Physical / logical core counts
  - Base & boost frequency
  - NUMA topology (nodes, cores-per-node)
  - CPU affinity control (pin workers to specific cores)
  - Architecture flags (SSE, AVX, AVX-512, NEON)
  - Dynamic worker-count recommendations for I/O-bound & CPU-bound work
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("pyaccelerate.cpu")


# ─────────────────────────────────────────────────────────────────────────
#  CPU Info Descriptor
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CPUInfo:
    """Immutable snapshot of detect CPU capabilities."""

    arch: str = ""                         # e.g. "x86_64", "aarch64"
    brand: str = ""                        # e.g. "Intel Core i7-12700K"
    physical_cores: int = 0
    logical_cores: int = 0
    frequency_mhz: float = 0.0            # base clock
    frequency_max_mhz: float = 0.0        # boost clock
    numa_nodes: int = 1
    cores_per_numa: List[int] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)  # ISA extensions

    @property
    def has_avx(self) -> bool:
        return any(f.startswith("avx") for f in self.flags)

    @property
    def has_avx512(self) -> bool:
        return any("avx512" in f for f in self.flags)

    @property
    def smt_ratio(self) -> float:
        """Logical-to-physical core ratio (2.0 = Hyper-Threading / SMT)."""
        if self.physical_cores <= 0:
            return 1.0
        return self.logical_cores / self.physical_cores

    def short_label(self) -> str:
        brand = self.brand or self.arch or "unknown"
        return f"{brand}  ({self.logical_cores}T / {self.physical_cores}C)"


# ─────────────────────────────────────────────────────────────────────────
#  Detection
# ─────────────────────────────────────────────────────────────────────────
_cached_info: Optional[CPUInfo] = None


def detect() -> CPUInfo:
    """Detect CPU capabilities. Result is cached after the first call."""
    global _cached_info
    if _cached_info is not None:
        return _cached_info

    arch = platform.machine() or ""
    logical = os.cpu_count() or 1
    brand = ""
    physical = logical
    freq_base = 0.0
    freq_max = 0.0
    flags: List[str] = []
    numa_nodes = 1
    cores_per_numa: List[int] = []

    # ── psutil (cross-platform) ──
    try:
        import psutil  # type: ignore[import-untyped]

        physical = psutil.cpu_count(logical=False) or logical
        freq = psutil.cpu_freq()
        if freq:
            freq_base = freq.current or 0.0
            freq_max = freq.max or freq_base
    except ImportError:
        pass

    # ── Platform-specific brand / flags ──
    system = platform.system()

    if system == "Windows":
        brand = _windows_brand()
        flags = _windows_flags()
    elif system == "Linux":
        brand, flags, numa_nodes, cores_per_numa = _linux_cpuinfo(logical)
    elif system == "Darwin":
        brand = _macos_brand()

    if not brand:
        brand = platform.processor() or arch

    if not cores_per_numa:
        # assume uniform distribution
        per_node = logical // max(numa_nodes, 1)
        cores_per_numa = [per_node] * numa_nodes

    info = CPUInfo(
        arch=arch,
        brand=brand,
        physical_cores=physical,
        logical_cores=logical,
        frequency_mhz=freq_base,
        frequency_max_mhz=freq_max,
        numa_nodes=numa_nodes,
        cores_per_numa=cores_per_numa,
        flags=flags,
    )
    _cached_info = info
    log.info("CPU: %s", info.short_label())
    return info


# ── Platform helpers ─────────────────────────────────────────────────────

def _windows_brand() -> str:
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0].strip()
    except Exception:
        pass
    return ""


def _windows_flags() -> List[str]:
    """Best-effort ISA flag detection on Windows via WMI + registry."""
    flags: List[str] = []
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Description"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            desc = r.stdout.strip().lower()
            for kw in ("sse", "sse2", "sse3", "sse4", "avx", "avx2", "avx512"):
                if kw in desc:
                    flags.append(kw)
    except Exception:
        pass
    return flags


def _linux_cpuinfo(
    logical: int,
) -> Tuple[str, List[str], int, List[int]]:
    """Parse /proc/cpuinfo & NUMA topology on Linux."""
    brand = ""
    flags: List[str] = []
    numa_nodes = 1
    cores_per_numa: List[int] = []

    try:
        text = open("/proc/cpuinfo").read()
        for line in text.splitlines():
            if line.startswith("model name") and not brand:
                brand = line.split(":", 1)[-1].strip()
            if line.startswith("flags"):
                raw = line.split(":", 1)[-1].strip().split()
                # keep interesting ones
                interesting = {"sse", "sse2", "sse3", "ssse3", "sse4_1", "sse4_2",
                               "avx", "avx2", "avx512f", "avx512bw", "avx512vl",
                               "fma", "aes", "neon", "asimd"}
                flags = sorted(set(raw) & interesting)
                break
    except Exception:
        pass

    try:
        import pathlib
        nodes = list(pathlib.Path("/sys/devices/system/node").glob("node[0-9]*"))
        numa_nodes = len(nodes) if nodes else 1
        for node in sorted(nodes):
            cpulist_file = node / "cpulist"
            if cpulist_file.exists():
                cpulist = cpulist_file.read_text().strip()
                count = _parse_cpulist_count(cpulist)
                cores_per_numa.append(count)
    except Exception:
        pass

    return brand, flags, numa_nodes, cores_per_numa


def _macos_brand() -> str:
    try:
        r = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return ""


def _parse_cpulist_count(cpulist: str) -> int:
    """Count CPUs in a Linux cpulist format string like '0-3,5,7-9'."""
    total = 0
    for part in cpulist.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            total += int(hi) - int(lo) + 1
        elif part.isdigit():
            total += 1
    return total


# ─────────────────────────────────────────────────────────────────────────
#  Affinity helpers
# ─────────────────────────────────────────────────────────────────────────

def pin_to_cores(cores: List[int]) -> bool:
    """Pin the current process to specific CPU cores. Returns True on success."""
    try:
        import psutil  # type: ignore[import-untyped]
        p = psutil.Process()
        p.cpu_affinity(cores)
        log.info("Pinned process to cores: %s", cores)
        return True
    except Exception as exc:
        log.warning("Failed to set CPU affinity: %s", exc)
        return False


def reset_affinity() -> bool:
    """Reset CPU affinity to all available cores."""
    try:
        import psutil  # type: ignore[import-untyped]
        p = psutil.Process()
        all_cores = list(range(os.cpu_count() or 1))
        p.cpu_affinity(all_cores)
        log.info("Reset affinity to all %d cores", len(all_cores))
        return True
    except Exception as exc:
        log.warning("Failed to reset CPU affinity: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────
#  Worker-count recommendations
# ─────────────────────────────────────────────────────────────────────────

def recommend_workers(
    io_bound: bool = True,
    ram_floor_gb: float = 4.0,
) -> int:
    """Recommend a thread/process worker count for the current host.

    Parameters
    ----------
    io_bound:
        True  → I/O-bound workload (threads spend most time waiting).
                Returns ``min(logical_cores × 3, 32)``.
        False → CPU-bound workload (threads use 100% CPU).
                Returns ``physical_cores`` (avoids SMT contention).
    ram_floor_gb:
        If available RAM is below this threshold, clamp the count further.
    """
    info = detect()
    ram_gb = _get_ram_gb()

    if io_bound:
        n = min(info.logical_cores * 3, 32)
    else:
        n = info.physical_cores

    if ram_gb < ram_floor_gb:
        n = min(n, max(2, info.physical_cores // 2))

    return max(1, n)


def recommend_io_workers() -> Tuple[int, int]:
    """Return ``(pull_workers, push_workers)`` tuned for I/O-bound transfers.

    Same heuristic used in adb-toolkit's TransferAccelerator:
      - pull: min(cores × 2, 16)
      - push: min(cores × 2, 12)
    Clamped further on low-RAM hosts (<4 GB).
    """
    info = detect()
    ram_gb = _get_ram_gb()

    pull = min(info.logical_cores * 2, 16)
    push = min(info.logical_cores * 2, 12)

    if ram_gb < 4:
        pull = min(pull, 6)
        push = min(push, 4)

    return max(2, pull), max(2, push)


def _get_ram_gb() -> float:
    try:
        import psutil  # type: ignore[import-untyped]
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return 8.0
