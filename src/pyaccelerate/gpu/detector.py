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
    # ARM mobile GPUs
    if any(k in nl for k in ("adreno", "qualcomm")):
        return "Qualcomm", False
    if any(k in nl for k in ("mali", "immortalis")):
        return "ARM", False
    if any(k in nl for k in ("xclipse", "samsung gpu")):
        return "Samsung", False
    if any(k in nl for k in ("powervr", "imagination")):
        return "Imagination", False
    if any(k in nl for k in ("maleoon",)):
        return "HiSilicon", False
    # SBC / IoT GPUs
    if any(k in nl for k in ("videocore", "vc4", "v3d")):
        return "Broadcom", False
    if any(k in nl for k in ("tegra", "jetson")):
        return "NVIDIA", False
    if any(k in nl for k in ("vivante", "galcore", "gc7000", "gc nano")):
        return "Vivante", False
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

        # ── ARM / Android GPU ──
        gpus.extend(_probe_arm_gpu(seen))

        # ── SBC / IoT GPU (VideoCore, Tegra, Vivante, etc.) ──
        gpus.extend(_probe_sbc_gpu(seen))

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


# ── ARM / Android GPU probe ─────────────────────────────────────────────

def _probe_arm_gpu(seen: set[str]) -> List[GPUDevice]:
    """Detect ARM mobile / embedded GPUs (Adreno, Mali, Immortalis, etc.).

    Uses:
      1. SoC database from android.py (reliable)
      2. SBC / IoT database (Raspberry Pi VideoCore, Jetson Tegra, etc.)
      3. Android sysfs: /sys/class/kgsl (Adreno), /sys/class/misc/mali0 (Mali)
      4. Vulkan via ``vulkaninfo`` (if installed in Termux)
      5. OpenCL already catches ARM GPUs in _probe_opencl
    """
    gpus: List[GPUDevice] = []

    try:
        from pyaccelerate.android import is_android, is_arm, get_soc_info
    except ImportError:
        return gpus

    if not is_arm():
        return gpus

    # ── Try SoC database first ──
    soc = get_soc_info()
    if soc and soc.gpu_name:
        key = soc.gpu_name.lower().strip()
        if key not in seen:
            vendor, _ = _vendor_from_name(soc.gpu_name)
            gpu = GPUDevice(
                name=soc.gpu_name,
                backend="none",
                vendor=vendor,
                compute_units=soc.gpu_cores,
                is_discrete=False,
            )
            # Try OpenCL on ARM to upgrade backend
            gpu = _try_arm_opencl_upgrade(gpu) or gpu
            gpus.append(gpu)
            seen.add(key)
            return gpus  # SoC DB is authoritative

    # ── Sysfs fallback: KGSL (Qualcomm Adreno) ──
    try:
        from pathlib import Path
        kgsl_path = Path("/sys/class/kgsl/kgsl-3d0")
        if kgsl_path.exists():
            gpu_name = "Adreno GPU"
            try:
                gpu_model = (kgsl_path / "gpu_model").read_text().strip()
                if gpu_model:
                    gpu_name = f"Adreno {gpu_model}"
            except Exception:
                pass
            key = gpu_name.lower().strip()
            if key not in seen:
                gpus.append(GPUDevice(
                    name=gpu_name, backend="none", vendor="Qualcomm",
                    is_discrete=False,
                ))
                seen.add(key)
    except Exception:
        pass

    # ── Sysfs fallback: Mali ──
    try:
        from pathlib import Path
        mali_paths = [
            Path("/sys/class/misc/mali0"),
            Path("/sys/devices/platform/mali-midgard"),
            Path("/sys/module/mali_kbase"),
        ]
        for mp in mali_paths:
            if mp.exists():
                gpu_name = "Mali GPU"
                # Try to read GPU ID from kernel
                try:
                    p = Path("/sys/module/mali_kbase/parameters/gpu_id")
                    if p.exists():
                        gpu_name = f"Mali ({p.read_text().strip()})"
                except Exception:
                    pass
                key = gpu_name.lower().strip()
                if key not in seen:
                    gpus.append(GPUDevice(
                        name=gpu_name, backend="none", vendor="ARM",
                        is_discrete=False,
                    ))
                    seen.add(key)
                break
    except Exception:
        pass

    # ── Vulkan fallback (Termux) ──
    if not gpus:
        try:
            r = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    ll = line.lower()
                    if "devicename" in ll or "device name" in ll:
                        gpu_name = line.split("=", 1)[-1].strip().strip('"')
                        if ":" in gpu_name:
                            gpu_name = gpu_name.split(":", 1)[-1].strip()
                        key = gpu_name.lower().strip()
                        if key and key not in seen:
                            vendor, _ = _vendor_from_name(gpu_name)
                            gpus.append(GPUDevice(
                                name=gpu_name, backend="vulkan",
                                vendor=vendor, is_discrete=False,
                            ))
                            seen.add(key)
                        break
        except Exception:
            pass

    return gpus


def _probe_sbc_gpu(seen: set[str]) -> List[GPUDevice]:
    """Detect SBC / IoT GPUs (VideoCore, Tegra CUDA, Vivante, etc.).

    Sources:
      1. SBC database from iot.py (board model → SoC → GPU name)
      2. Jetson Tegra — already found via CUDA probe, but add fallback
      3. VideoCore (Raspberry Pi) via /dev/vchiq or vcgencmd
      4. Vivante (NXP i.MX) via /dev/galcore
    """
    gpus: List[GPUDevice] = []

    try:
        from pyaccelerate.iot import is_sbc, detect_sbc, is_jetson
    except ImportError:
        return gpus

    if not is_sbc():
        return gpus

    sbc = detect_sbc()
    if not sbc or not sbc.gpu_name:
        return gpus

    key = sbc.gpu_name.lower().strip()
    if key in seen:
        return gpus

    vendor = sbc.soc_vendor
    cuda_cores = sbc.gpu_cuda_cores
    backend = "none"

    # Jetson: CUDA is handled by _probe_cuda, but mark as cuda if available
    if sbc.family == "jetson" and cuda_cores > 0:
        try:
            import cupy  # type: ignore[import-untyped]
            backend = "cuda"
        except ImportError:
            pass

    # VideoCore: check for vcgencmd
    if "videocore" in key:
        vendor = "Broadcom"
        try:
            from pathlib import Path
            if Path("/dev/vchiq").exists():
                backend = "videocore"
        except Exception:
            pass

    # Vivante (NXP i.MX): check for galcore driver
    if "vivante" in key:
        vendor = "Vivante"
        try:
            from pathlib import Path
            if Path("/dev/galcore").exists():
                backend = "vivante"
        except Exception:
            pass

    gpu = GPUDevice(
        name=sbc.gpu_name,
        backend=backend,
        vendor=vendor,
        compute_units=cuda_cores,
        is_discrete=False,
    )
    # Try OpenCL upgrade
    gpu = _try_arm_opencl_upgrade(gpu) or gpu
    gpus.append(gpu)
    seen.add(key)

    return gpus


def _try_arm_opencl_upgrade(gpu: GPUDevice) -> Optional[GPUDevice]:
    """Try to upgrade an ARM GPU from 'none' to 'opencl' backend."""
    try:
        import pyopencl as cl  # type: ignore[import-untyped]
        for plat in cl.get_platforms():
            for dev in plat.get_devices(device_type=cl.device_type.GPU):
                dev_name = dev.name.strip()
                # Check if this CL device matches our GPU
                if _gpu_names_match(gpu.name, dev_name):
                    try:
                        cus = dev.max_compute_units
                    except Exception:
                        cus = gpu.compute_units
                    return GPUDevice(
                        name=dev_name,
                        backend="opencl",
                        vendor=gpu.vendor,
                        memory_bytes=dev.global_mem_size,
                        compute_units=cus,
                        is_discrete=False,
                        _module=cl,
                    )
    except Exception:
        pass
    return None


def _gpu_names_match(name_a: str, name_b: str) -> bool:
    """Check if two GPU name strings likely refer to the same device."""
    a, b = name_a.lower(), name_b.lower()
    # Exact substring match
    if a in b or b in a:
        return True
    # Check key identifiers (e.g., "adreno 750" vs "Qualcomm Adreno 750")
    for token in ("adreno", "mali", "immortalis", "xclipse", "powervr", "maleoon",
                   "videocore", "tegra", "vivante"):
        if token in a and token in b:
            return True
    return False


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
            # ARM/Android: no lspci — use getprop or device-tree
            try:
                from pyaccelerate.android import is_android, get_soc_info
                if is_android():
                    soc = get_soc_info()
                    if soc and soc.gpu_name:
                        names.append(soc.gpu_name)
                    if not names:
                        # Try getprop for GPU hints
                        r2 = subprocess.run(
                            ["getprop", "ro.hardware.egl"],
                            capture_output=True, text=True, timeout=3,
                        )
                        if r2.returncode == 0 and r2.stdout.strip():
                            names.append(r2.stdout.strip())
            except ImportError:
                pass

            if not names:
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
    # ARM GPUs — suggest OpenCL or Vulkan
    for g in gpus:
        vl = g.vendor.lower()
        if vl in ("qualcomm", "arm", "samsung", "imagination", "hisilicon"):
            hints.append("pip install pyopencl  # ARM OpenCL")
    # SBC / IoT GPUs
    for g in gpus:
        vl = g.vendor.lower()
        nl = g.name.lower()
        if "broadcom" in vl or "videocore" in nl:
            hints.append("pip install pyopencl  # VideoCore OpenCL (RPi)")
        elif "vivante" in vl or "vivante" in nl:
            hints.append("pip install pyopencl  # Vivante OpenCL (i.MX)")
    if hints:
        return "Install GPU support:  " + "  or  ".join(sorted(set(hints)))
    return ""
