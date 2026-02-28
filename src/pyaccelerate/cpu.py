"""
pyaccelerate.cpu — CPU detection, topology and affinity utilities.

Provides runtime introspection of the host CPU capabilities:
  - Physical / logical core counts
  - Base & boost frequency
  - NUMA topology (nodes, cores-per-node)
  - CPU affinity control (pin workers to specific cores)
  - Architecture flags (SSE, AVX, AVX-512, NEON, ASIMD, SVE)
  - ARM big.LITTLE / DynamIQ cluster detection
  - Android / Termux awareness
  - Dynamic worker-count recommendations for I/O-bound & CPU-bound work
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("pyaccelerate.cpu")


# ─────────────────────────────────────────────────────────────────────────
#  CPU Info Descriptor
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CPUInfo:
    """Immutable snapshot of detected CPU capabilities."""

    arch: str = ""                         # e.g. "x86_64", "aarch64", "armv7l"
    brand: str = ""                        # e.g. "Intel Core i7-12700K", "Snapdragon 8 Gen 3"
    physical_cores: int = 0
    logical_cores: int = 0
    frequency_mhz: float = 0.0            # base clock
    frequency_max_mhz: float = 0.0        # boost clock
    numa_nodes: int = 1
    cores_per_numa: List[int] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)  # ISA extensions
    is_arm: bool = False                   # ARM architecture (AArch64 / ARMv7)
    arm_clusters: Dict[str, List[int]] = field(default_factory=dict)  # big.LITTLE
    soc_name: str = ""                     # SoC name on ARM devices
    is_android: bool = False               # Running on Android / Termux
    is_sbc: bool = False                   # Running on a Single-Board Computer

    @property
    def has_avx(self) -> bool:
        return any(f.startswith("avx") for f in self.flags)

    @property
    def has_avx512(self) -> bool:
        return any("avx512" in f for f in self.flags)

    @property
    def has_neon(self) -> bool:
        """True if ARM NEON / ASIMD is available."""
        return any(f in ("neon", "asimd") for f in self.flags)

    @property
    def has_sve(self) -> bool:
        """True if ARM SVE / SVE2 is available."""
        return any(f.startswith("sve") for f in self.flags)

    @property
    def big_cores(self) -> int:
        """Count of performance (big) cores on ARM big.LITTLE."""
        if not self.arm_clusters:
            return 0
        # Biggest cluster by max freq or known ordering
        clusters = list(self.arm_clusters.values())
        if len(clusters) == 1:
            return len(clusters[0])
        # First cluster in sorted order is typically "big"
        return len(clusters[0]) if clusters else 0

    @property
    def little_cores(self) -> int:
        """Count of efficiency (LITTLE) cores on ARM big.LITTLE."""
        if not self.arm_clusters or len(self.arm_clusters) <= 1:
            return 0
        clusters = list(self.arm_clusters.values())
        return len(clusters[-1])

    @property
    def smt_ratio(self) -> float:
        """Logical-to-physical core ratio (2.0 = Hyper-Threading / SMT)."""
        if self.physical_cores <= 0:
            return 1.0
        return self.logical_cores / self.physical_cores

    def short_label(self) -> str:
        brand = self.brand or self.soc_name or self.arch or "unknown"
        suffix = f"({self.logical_cores}T / {self.physical_cores}C)"
        if self.arm_clusters and len(self.arm_clusters) > 1:
            cluster_desc = "+".join(
                f"{len(cpus)}{name.split('-')[-1] if '-' in name else name[:3]}"
                for name, cpus in self.arm_clusters.items()
            )
            suffix = f"({self.logical_cores}C {cluster_desc})"
        return f"{brand}  {suffix}"


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
        try:
            freq = psutil.cpu_freq()
        except (PermissionError, OSError):
            freq = None
        if freq:
            freq_base = freq.current or 0.0
            freq_max = freq.max or freq_base
    except ImportError:
        pass

    # ── ARM detection ──
    _is_arm = _check_is_arm(arch)
    _is_android_flag = False
    arm_clusters: Dict[str, List[int]] = {}
    soc_name = ""

    try:
        from pyaccelerate.android import is_android as _is_android_fn, is_arm as _is_arm_fn
        _is_android_flag = _is_android_fn()
    except Exception:
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

    # ── ARM-specific enrichment ──
    if _is_arm:
        arm_brand, arm_flags, arm_clusters, arm_freq_max, soc_name = _arm_detect(
            arch, brand, flags, _is_android_flag
        )
        if arm_brand and (not brand or brand == arch):
            brand = arm_brand
        if arm_flags:
            flags = arm_flags
        if arm_freq_max > freq_max:
            freq_max = arm_freq_max

    if not brand:
        brand = platform.processor() or arch

    if not cores_per_numa:
        # assume uniform distribution
        per_node = logical // max(numa_nodes, 1)
        cores_per_numa = [per_node] * numa_nodes

    # ── SBC / IoT detection ──
    _is_sbc = False
    try:
        from pyaccelerate.iot import is_sbc as _check_sbc
        _is_sbc = _check_sbc()
    except ImportError:
        pass

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
        is_arm=_is_arm,
        arm_clusters=arm_clusters,
        soc_name=soc_name,
        is_android=_is_android_flag,
        is_sbc=_is_sbc,
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


def _check_is_arm(arch: str) -> bool:
    """Check if architecture is ARM."""
    al = arch.lower()
    return any(arm in al for arm in ("aarch64", "arm64", "armv7", "armv8", "arm"))


def _arm_detect(
    arch: str,
    current_brand: str,
    current_flags: List[str],
    is_android: bool,
) -> tuple:
    """ARM-specific CPU detection.

    Returns (brand, flags, clusters, max_freq_mhz, soc_name).
    """
    brand = current_brand
    flags = current_flags
    clusters: Dict[str, List[int]] = {}
    max_freq = 0.0
    soc_name = ""

    try:
        from pyaccelerate.android import (
            detect_arm_cores, detect_big_little, get_arm_features,
            get_soc_info, get_cpu_freq_info, get_device_info,
        )

        # ARM features (NEON, SVE, etc)
        arm_feat = get_arm_features()
        if arm_feat:
            flags = arm_feat

        # big.LITTLE clusters
        bl = detect_big_little()
        if bl:
            clusters = bl

        # SoC identification
        soc = get_soc_info()
        if soc:
            soc_name = soc.name
            brand = soc.name

        # Max frequency from sysfs
        freq_info = get_cpu_freq_info()
        for cpu_name, freqs in freq_info.items():
            mf = freqs.get("cpuinfo_max_freq", freqs.get("scaling_max_freq", 0.0))
            if mf > max_freq:
                max_freq = mf

        # Android getprop brand
        if is_android and not soc_name:
            dev = get_device_info()
            if dev.get("soc"):
                soc_name = dev["soc"]
                brand = soc_name
            elif dev.get("hardware"):
                brand = dev["hardware"]

    except Exception as exc:
        log.debug("ARM detection partial: %s", exc)

    return brand, flags, clusters, max_freq, soc_name


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

    On ARM Android devices with big.LITTLE, CPU-bound work defaults to the
    count of **performance** (big) cores only to avoid thermal throttling.

    On SBC / IoT devices with limited RAM, the count is further clamped
    using ``iot.recommend_iot_workers()``.
    """
    info = detect()
    ram_gb = _get_ram_gb()

    # SBC / IoT: use memory-aware recommendation
    if info.is_sbc:
        try:
            from pyaccelerate.iot import recommend_iot_workers
            return max(1, recommend_iot_workers(io_bound=io_bound))
        except ImportError:
            pass

    if io_bound:
        n = min(info.logical_cores * 3, 32)
        # ARM devices typically have less RAM — lower the cap
        if info.is_android:
            n = min(n, 16)
    else:
        # On big.LITTLE, prefer only performance cores for CPU-bound
        if info.arm_clusters and len(info.arm_clusters) > 1:
            # Use big + mid cores, exclude LITTLE for CPU-bound
            clusters = list(info.arm_clusters.values())
            n = sum(len(c) for c in clusters[:-1])  # all but smallest cluster
            n = max(n, 2)
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
