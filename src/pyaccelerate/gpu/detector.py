"""
pyaccelerate.gpu.detector — Multi-vendor, multi-backend GPU enumeration.

Enumerates ALL GPUs on the system across:
  - CuPy / CUDA  (NVIDIA)
  - PyOpenCL      (NVIDIA, AMD, Intel)
  - Intel oneAPI / dpctl (Intel, including Arc)
  - OS-level fallback (display-only, no compute)

GPUs are ranked by a composite *score* (VRAM + compute units + discrete bonus).
Detection runs once and is cached; call ``reset_cache()`` to force re-scan.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("pyaccelerate.gpu.detector")


# ═══════════════════════════════════════════════════════════════════════════
#  GPU Device Descriptor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GPUDevice:
    """Represents one detected GPU compute device."""

    name: str = ""
    backend: str = ""          # "cuda", "opencl", "intel", "none"
    vendor: str = ""           # "NVIDIA", "Intel", "AMD", "unknown"
    memory_bytes: int = 0      # VRAM (global memory) in bytes
    compute_units: int = 0     # SMs / CUs / EUs
    is_discrete: bool = False  # discrete vs integrated
    _module: Any = None        # runtime handle (cupy, pyopencl ctx, dpctl device)
    _index: int = 0            # device ordinal in its backend

    @property
    def memory_gb(self) -> float:
        return self.memory_bytes / (1024 ** 3) if self.memory_bytes else 0.0

    @property
    def score(self) -> int:
        """Composite power score for ranking. Discrete GPUs get a large bonus."""
        s = self.memory_bytes // (1024 * 1024)  # MB of VRAM
        s += self.compute_units * 50
        if self.is_discrete:
            s += 100_000
        return s

    @property
    def usable(self) -> bool:
        """True if a compute backend is available (not just OS-level name)."""
        return self.backend != "none"

    def short_label(self) -> str:
        mem = f"{self.memory_gb:.1f} GB" if self.memory_bytes else "?"
        return f"{self.name} ({self.backend.upper()}, {mem})"

    def as_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "backend": self.backend,
            "vendor": self.vendor,
            "memory": f"{self.memory_gb:.1f} GB",
            "compute_units": str(self.compute_units),
            "discrete": str(self.is_discrete),
            "score": str(self.score),
            "usable": str(self.usable),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Cache & lock
# ═══════════════════════════════════════════════════════════════════════════

_all_gpus: List[GPUDevice] = []
_best_gpu: Optional[GPUDevice] = None
_detected = False
_detect_lock = threading.Lock()


def reset_cache() -> None:
    """Force re-detection on next call to ``detect_all()``."""
    global _all_gpus, _best_gpu, _detected
    with _detect_lock:
        _all_gpus = []
        _best_gpu = None
        _detected = False


# ═══════════════════════════════════════════════════════════════════════════
#  Vendor heuristic
# ═══════════════════════════════════════════════════════════════════════════

def _vendor_from_name(name: str) -> Tuple[str, bool]:
    """Guess vendor and discrete flag from device name string."""
    nl = name.lower()
    if any(k in nl for k in ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla", "a100", "h100")):
        return "NVIDIA", True
    if any(k in nl for k in ("radeon", "amd", "rx ", "vega", "instinct")):
        return "AMD", True
    if any(k in nl for k in ("intel", "uhd", "iris", "arc")):
        is_discrete = "arc" in nl
        return "Intel", is_discrete
    if any(k in nl for k in ("apple", "m1", "m2", "m3", "m4")):
        return "Apple", True
    return "unknown", False


# ═══════════════════════════════════════════════════════════════════════════
#  Detection — enumerate all backends
# ═══════════════════════════════════════════════════════════════════════════

def detect_all() -> List[GPUDevice]:
    """Enumerate ALL GPUs from every available compute backend.

    Returns a list sorted by ``score`` (best first). Thread-safe, cached.
    """
    global _all_gpus, _best_gpu, _detected
    if _detected:
        return _all_gpus
    with _detect_lock:
        if _detected:
            return _all_gpus

        gpus: List[GPUDevice] = []
        seen: set[str] = set()

        # ── CuPy / CUDA ──
        gpus.extend(_probe_cuda(seen))

        # ── PyOpenCL ──
        gpus.extend(_probe_opencl(seen))

        # ── Intel oneAPI (dpctl) ──
        gpus.extend(_probe_intel(seen))

        # ── OS-level fallback ──
        if not gpus:
            for hw_name in _detect_os_gpu_names():
                vendor, discrete = _vendor_from_name(hw_name)
                gpus.append(GPUDevice(
                    name=hw_name, backend="none", vendor=vendor,
                    is_discrete=discrete,
                ))

        gpus.sort(key=lambda g: g.score, reverse=True)
        _all_gpus = gpus
        _best_gpu = gpus[0] if gpus else None

        if _best_gpu and _best_gpu.usable:
            log.info(
                "Best GPU: %s (%s, score=%d). Total: %d",
                _best_gpu.name, _best_gpu.backend, _best_gpu.score, len(gpus),
            )
        elif gpus:
            log.info("GPU(s) detected but no compute library: %s",
                     ", ".join(g.name for g in gpus))
        else:
            log.info("No GPU detected")

        _detected = True
        return _all_gpus


# ── Backend probes ──────────────────────────────────────────────────────

def _probe_cuda(seen: set[str]) -> List[GPUDevice]:
    """Probe CUDA devices via CuPy."""
    gpus: List[GPUDevice] = []
    try:
        import cupy as cp  # type: ignore[import-untyped]
        n = cp.cuda.runtime.getDeviceCount()
        for i in range(n):
            try:
                props = cp.cuda.runtime.getDeviceProperties(i)
                dev_name = props["name"]
                if isinstance(dev_name, bytes):
                    dev_name = dev_name.decode()
                mem = props.get("totalGlobalMem", 0)
                sms = props.get("multiProcessorCount", 0)
                vendor, discrete = _vendor_from_name(dev_name)
                gpus.append(GPUDevice(
                    name=dev_name, backend="cuda", vendor=vendor,
                    memory_bytes=mem, compute_units=sms,
                    is_discrete=discrete, _module=cp, _index=i,
                ))
                seen.add(dev_name.lower().strip())
            except Exception:
                pass
    except Exception as exc:
        log.debug("CuPy/CUDA not available: %s", exc)
    return gpus


def _probe_opencl(seen: set[str]) -> List[GPUDevice]:
    """Probe GPU devices via PyOpenCL."""
    gpus: List[GPUDevice] = []
    try:
        import pyopencl as cl  # type: ignore[import-untyped]
        for plat in cl.get_platforms():
            try:
                for dev in plat.get_devices(device_type=cl.device_type.GPU):
                    dev_name = dev.name.strip()
                    key = dev_name.lower().strip()
                    if key in seen:
                        continue
                    vendor, discrete = _vendor_from_name(dev_name)
                    try:
                        cus = dev.max_compute_units
                    except Exception:
                        cus = 0
                    gpus.append(GPUDevice(
                        name=dev_name, backend="opencl", vendor=vendor,
                        memory_bytes=dev.global_mem_size,
                        compute_units=cus, is_discrete=discrete,
                        _module=cl, _index=0,
                    ))
                    seen.add(key)
            except Exception:
                continue
    except Exception as exc:
        log.debug("PyOpenCL not available: %s", exc)
    return gpus


def _probe_intel(seen: set[str]) -> List[GPUDevice]:
    """Probe GPU devices via Intel oneAPI (dpctl)."""
    gpus: List[GPUDevice] = []
    try:
        import dpctl  # type: ignore[import-untyped]
        for dev in dpctl.get_devices():
            if dev.device_type.name != "gpu":
                continue
            dev_name = dev.name
            key = dev_name.lower().strip()
            if key in seen:
                continue
            vendor, discrete = _vendor_from_name(dev_name)
            try:
                mem = dev.global_mem_size
            except Exception:
                mem = 0
            gpus.append(GPUDevice(
                name=dev_name, backend="intel", vendor=vendor,
                memory_bytes=mem, is_discrete=discrete,
                _module=dpctl, _index=0,
            ))
            seen.add(key)
    except Exception as exc:
        log.debug("dpctl/oneAPI not available: %s", exc)
    return gpus


# ── OS-level fallback ───────────────────────────────────────────────────

def _detect_os_gpu_names() -> List[str]:
    """Detect GPU names at OS level (no compute capability)."""
    names: List[str] = []
    try:
        if platform.system() == "Windows":
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().splitlines():
                    line = line.strip()
                    if line:
                        names.append(line)
        elif platform.system() == "Darwin":
            r = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if "Chipset Model" in line:
                        names.append(line.split(":", 1)[-1].strip())
        else:
            r = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=10,
            )
            for line in r.stdout.splitlines():
                if "VGA" in line or "3D" in line or "Display" in line:
                    names.append(line.split(":", 2)[-1].strip())
    except Exception:
        pass
    return names


# ═══════════════════════════════════════════════════════════════════════════
#  Public convenience API
# ═══════════════════════════════════════════════════════════════════════════

def gpu_available() -> bool:
    """True if at least one GPU with a compute backend exists."""
    return any(g.usable for g in detect_all())


def best_gpu() -> Optional[GPUDevice]:
    """Return the highest-scored usable GPU, or None."""
    detect_all()
    if _best_gpu and _best_gpu.usable:
        return _best_gpu
    return None


def get_gpu_info() -> Dict[str, str]:
    """Info dict for the best GPU (or CPU fallback)."""
    gpus = detect_all()
    top = gpus[0] if gpus else None
    if top is None or not top.usable:
        hw = top.name if top else "N/A"
        return {
            "available": "false",
            "backend": "cpu",
            "device": hw or "N/A",
            "note": "No GPU compute library — using CPU",
        }
    return {
        "available": "true",
        "backend": top.backend,
        "device": top.name,
        "memory": f"{top.memory_gb:.1f} GB",
        "vendor": top.vendor,
        "score": str(top.score),
    }


def get_all_gpus_info() -> List[Dict[str, str]]:
    """Info dicts for ALL detected GPUs (best-first)."""
    return [g.as_dict() for g in detect_all()]


def get_install_hint() -> str:
    """Suggest pip install commands based on detected hardware."""
    gpus = detect_all()
    usable = [g for g in gpus if g.usable]
    if usable:
        return ""
    if not gpus:
        return "No GPU detected. CPU multi-threading will be used."
    hints: list[str] = []
    for g in gpus:
        vl = g.vendor.lower()
        if "nvidia" in vl:
            hints.append("pip install cupy-cuda12x")
        elif "intel" in vl:
            hints.append("pip install pyopencl")
        elif "amd" in vl:
            hints.append("pip install pyopencl")
    if hints:
        return "Install GPU support:  " + "  or  ".join(sorted(set(hints)))
    return ""
