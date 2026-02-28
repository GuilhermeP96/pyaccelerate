"""
pyaccelerate.npu.detector — NPU hardware detection across all vendors.

Detection strategy (ordered by priority):
  1. **OpenVINO** — ``openvino.runtime.Core().get_property("NPU", ...)``
  2. **ONNX Runtime** — check available Execution Providers (DML, QNN, OpenVINO)
  3. **intel-npu-acceleration-library** — direct Intel NPU access
  4. **DirectML** (Windows) — ``dxcore`` enumeration for NPU class devices
  5. **OS-level** — WMI / sysfs / system_profiler fallback (display-only)

Each detected NPU is represented as an ``NPUDevice`` dataclass and ranked
by a composite score.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("pyaccelerate.npu.detector")


# ═══════════════════════════════════════════════════════════════════════════
#  NPU Device Descriptor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NPUDevice:
    """Represents one detected NPU compute device."""

    name: str = ""
    vendor: str = ""               # "Intel", "AMD", "Qualcomm", "Apple", "unknown"
    backend: str = ""              # "openvino", "onnxrt-dml", "onnxrt-qnn", "intel-npu", "coreml", "none"
    tops: float = 0.0              # Tera Operations Per Second (advertised)
    driver_version: str = ""
    _module: Any = None            # runtime handle
    _ep_name: str = ""             # ONNX Runtime Execution Provider name

    @property
    def usable(self) -> bool:
        """True if a compute framework can target this NPU."""
        return self.backend not in ("none", "")

    @property
    def score(self) -> int:
        """Composite ranking score.  Higher = better."""
        s = int(self.tops * 1000)  # TOPS → base score
        # Backend preference bonus
        backend_bonus = {
            "openvino": 500,
            "intel-npu": 400,
            "onnxrt-qnn": 300,
            "onnxrt-dml": 200,
            "coreml": 300,
        }
        s += backend_bonus.get(self.backend, 0)
        return max(s, 1)

    def short_label(self) -> str:
        tops_str = f"{self.tops:.1f} TOPS" if self.tops else "? TOPS"
        return f"{self.name} ({self.backend}, {tops_str})"

    def as_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "backend": self.backend,
            "tops": f"{self.tops:.1f}",
            "driver_version": self.driver_version,
            "score": str(self.score),
            "usable": str(self.usable),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Known NPU TOPS ratings (approximate, advertised)
# ═══════════════════════════════════════════════════════════════════════════

_KNOWN_TOPS: Dict[str, float] = {
    # Intel
    "meteor lake": 10.0,
    "arrow lake": 13.0,
    "lunar lake": 48.0,
    "panther lake": 60.0,
    # AMD
    "ryzen ai 9 hx 370": 50.0,
    "ryzen ai 9 365": 50.0,
    "ryzen ai 9 hx 375": 55.0,
    "ryzen ai 300": 50.0,
    "ryzen 7 8845hs": 16.0,
    "ryzen 9 8945hs": 16.0,
    "ryzen 7 8840u": 16.0,
    "ryzen 5 8640u": 16.0,
    "phoenix": 10.0,
    "hawk point": 16.0,
    "strix point": 50.0,
    # Qualcomm
    "snapdragon x elite": 45.0,
    "snapdragon x plus": 45.0,
    "snapdragon x": 45.0,
    "snapdragon 8 gen 3": 73.0,
    "snapdragon 8 gen 4": 80.0,
    "snapdragon 8 elite": 80.0,
    "snapdragon 8 gen 2": 36.0,
    "snapdragon 8 gen 1": 27.0,
    "snapdragon 888": 26.0,
    "snapdragon 865": 15.0,
    "snapdragon 7+ gen 3": 45.0,
    "snapdragon 7 gen 3": 20.0,
    "snapdragon 6 gen 1": 12.0,
    "hexagon npu": 45.0,
    "hexagon 780": 26.0,
    "hexagon 698": 15.0,
    # Samsung Exynos
    "exynos 2500": 50.0,
    "exynos 2200": 34.7,
    "exynos 2100": 26.0,
    "exynos 1380": 5.2,
    "exynos 990": 10.0,
    "samsung npu": 26.0,
    # Google Tensor
    "tensor g4": 35.0,
    "tensor g3": 30.0,
    "tensor g2": 22.0,
    "tensor g1": 18.0,
    "google tpu": 18.0,
    # MediaTek Dimensity
    "dimensity 9300": 46.0,
    "dimensity 9200": 35.0,
    "dimensity 9000": 25.0,
    "dimensity 8300": 20.0,
    "dimensity 1200": 9.0,
    "dimensity 1100": 7.0,
    "dimensity 900": 5.0,
    "mediatek apu 790": 46.0,
    "mediatek apu 690": 35.0,
    "mediatek apu 590": 25.0,
    "mediatek apu": 9.0,
    # HiSilicon Kirin
    "kirin 9010": 25.0,
    "kirin 9000": 12.0,
    "da vinci npu": 12.0,
    # Apple
    "m1": 11.0,
    "m1 pro": 11.0,
    "m1 max": 11.0,
    "m1 ultra": 22.0,
    "m2": 15.8,
    "m2 pro": 15.8,
    "m2 max": 15.8,
    "m2 ultra": 31.6,
    "m3": 18.0,
    "m3 pro": 18.0,
    "m3 max": 18.0,
    "m4": 38.0,
    "m4 pro": 38.0,
    "m4 max": 38.0,
}


def _estimate_tops(name: str) -> float:
    """Estimate TOPS from device/CPU name using known hardware DB."""
    nl = name.lower()
    for pattern, tops in _KNOWN_TOPS.items():
        if pattern in nl:
            return tops
    return 0.0


def _vendor_from_name(name: str) -> str:
    """Guess NPU vendor from device name."""
    nl = name.lower()
    if any(k in nl for k in ("intel", "meteor", "arrow", "lunar", "panther")):
        return "Intel"
    if any(k in nl for k in ("amd", "ryzen", "xdna", "phoenix", "hawk", "strix")):
        return "AMD"
    if any(k in nl for k in ("qualcomm", "snapdragon", "hexagon", "qnn")):
        return "Qualcomm"
    if any(k in nl for k in ("apple", "neural engine", "ane")):
        return "Apple"
    if any(k in nl for k in ("mediatek", "dimensity", "apu 790", "apu 690", "apu 590", "apu 3")):
        return "MediaTek"
    if any(k in nl for k in ("samsung", "exynos", "xclipse")):
        return "Samsung"
    if any(k in nl for k in ("google", "tensor", "tpu")):
        return "Google"
    if any(k in nl for k in ("hisilicon", "kirin", "da vinci", "davinci")):
        return "HiSilicon"
    if any(k in nl for k in ("unisoc",)):
        return "Unisoc"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
#  Cache
# ═══════════════════════════════════════════════════════════════════════════

_all_npus: List[NPUDevice] = []
_best_npu: Optional[NPUDevice] = None
_detected = False
_detect_lock = threading.Lock()


def reset_cache() -> None:
    """Force re-detection on next call."""
    global _all_npus, _best_npu, _detected
    with _detect_lock:
        _all_npus = []
        _best_npu = None
        _detected = False


# ═══════════════════════════════════════════════════════════════════════════
#  Main detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_all() -> List[NPUDevice]:
    """Enumerate all NPU devices.  Sorted by score (best first).  Cached."""
    global _all_npus, _best_npu, _detected
    if _detected:
        return _all_npus
    with _detect_lock:
        if _detected:
            return _all_npus

        npus: List[NPUDevice] = []
        seen: set[str] = set()

        # 1. OpenVINO NPU plugin
        npus.extend(_probe_openvino(seen))

        # 2. ONNX Runtime Execution Providers
        npus.extend(_probe_onnxrt(seen))

        # 3. intel-npu-acceleration-library
        npus.extend(_probe_intel_npu_lib(seen))

        # 4. ARM / Android NPU (mobile SoCs)
        npus.extend(_probe_arm_npu(seen))

        # 5. OS-level fallback
        if not npus:
            npus.extend(_probe_os_level(seen))

        npus.sort(key=lambda n: n.score, reverse=True)
        _all_npus = npus
        _best_npu = npus[0] if npus else None

        if _best_npu and _best_npu.usable:
            log.info("Best NPU: %s (score=%d)", _best_npu.short_label(), _best_npu.score)
        elif npus:
            log.info("NPU detected but no compute framework: %s",
                     ", ".join(n.name for n in npus))
        else:
            log.debug("No NPU detected")

        _detected = True
        return _all_npus


# ── Backend probes ──────────────────────────────────────────────────────

def _probe_openvino(seen: set[str]) -> List[NPUDevice]:
    """Detect NPU via OpenVINO Runtime."""
    npus: List[NPUDevice] = []
    try:
        from openvino import Core  # type: ignore[import-untyped]
        core = Core()
        devices = core.available_devices
        if "NPU" not in devices:
            return npus

        try:
            full_name = core.get_property("NPU", "FULL_DEVICE_NAME")
        except Exception:
            full_name = "Intel NPU"

        try:
            driver = core.get_property("NPU", "NPU_DRIVER_VERSION")
        except Exception:
            driver = ""

        vendor = _vendor_from_name(full_name)
        tops = _estimate_tops(full_name)

        # Try to get TOPS from CPU name if NPU name doesn't match
        if tops == 0.0:
            cpu_name = _get_cpu_name()
            tops = _estimate_tops(cpu_name)

        npu = NPUDevice(
            name=full_name,
            vendor=vendor,
            backend="openvino",
            tops=tops,
            driver_version=driver,
            _module=core,
        )
        npus.append(npu)
        seen.add(full_name.lower().strip())
        log.debug("OpenVINO NPU found: %s", full_name)

    except Exception as exc:
        log.debug("OpenVINO NPU not available: %s", exc)
    return npus


def _probe_onnxrt(seen: set[str]) -> List[NPUDevice]:
    """Detect NPU-capable ONNX Runtime Execution Providers."""
    npus: List[NPUDevice] = []
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
        available_eps = ort.get_available_providers()

        # NPU-capable EPs, in preference order
        npu_eps = [
            ("OpenVINOExecutionProvider", "openvino", "Intel"),
            ("QNNExecutionProvider", "onnxrt-qnn", "Qualcomm"),
            ("DmlExecutionProvider", "onnxrt-dml", ""),  # vendor detected from HW
        ]

        for ep_name, backend, default_vendor in npu_eps:
            if ep_name not in available_eps:
                continue

            # Skip if we already found this via native probe
            key = f"onnxrt-{ep_name}".lower()
            if key in seen:
                continue

            name = f"NPU via {ep_name}"
            vendor = default_vendor or _detect_onnxrt_vendor()
            cpu_name = _get_cpu_name()
            tops = _estimate_tops(cpu_name)

            npu = NPUDevice(
                name=name,
                vendor=vendor,
                backend=backend,
                tops=tops,
                _module=ort,
                _ep_name=ep_name,
            )
            npus.append(npu)
            seen.add(key)
            log.debug("ONNX Runtime NPU EP found: %s", ep_name)

    except Exception as exc:
        log.debug("ONNX Runtime not available: %s", exc)
    return npus


def _probe_intel_npu_lib(seen: set[str]) -> List[NPUDevice]:
    """Detect via intel-npu-acceleration-library."""
    npus: List[NPUDevice] = []
    try:
        import intel_npu_acceleration_library as npu_lib  # type: ignore[import-untyped]

        key = "intel-npu-accel"
        if key in seen:
            return npus

        cpu_name = _get_cpu_name()
        tops = _estimate_tops(cpu_name)

        npu = NPUDevice(
            name=f"Intel NPU ({cpu_name})" if cpu_name else "Intel NPU",
            vendor="Intel",
            backend="intel-npu",
            tops=tops,
            _module=npu_lib,
        )
        npus.append(npu)
        seen.add(key)
        log.debug("intel-npu-acceleration-library detected")

    except Exception as exc:
        log.debug("intel-npu-acceleration-library not available: %s", exc)
    return npus


def _probe_os_level(seen: set[str]) -> List[NPUDevice]:
    """OS-level NPU detection (display only, no compute)."""
    npus: List[NPUDevice] = []
    system = platform.system()

    if system == "Windows":
        npus.extend(_probe_windows_npu(seen))
    elif system == "Darwin":
        npus.extend(_probe_macos_npu(seen))
    elif system == "Linux":
        npus.extend(_probe_linux_npu(seen))

    return npus


def _probe_arm_npu(seen: set[str]) -> List[NPUDevice]:
    """Detect ARM mobile/embedded NPUs via SoC database and Android APIs.

    Sources:
      1. SoC database from android.py (Hexagon, Samsung NPU, Tensor TPU, etc.)
      2. Android NNAPI availability check
      3. TFLite delegate probing
      4. /dev/accel on Android
    """
    npus: List[NPUDevice] = []

    try:
        from pyaccelerate.android import is_arm, is_android, get_soc_info
    except ImportError:
        return npus

    if not is_arm():
        return npus

    # ── SoC database NPU ──
    soc = get_soc_info()
    if soc and soc.npu_name and soc.npu_name != "N/A":
        key = soc.npu_name.lower().strip()
        if key not in seen:
            vendor = _vendor_from_name(soc.npu_name) or soc.vendor
            tops = soc.npu_tops if soc.npu_tops > 0 else _estimate_tops(soc.name)
            backend = _detect_arm_npu_backend(soc)
            npus.append(NPUDevice(
                name=f"{soc.npu_name} ({soc.name})",
                vendor=vendor,
                backend=backend,
                tops=tops,
            ))
            seen.add(key)

    # ── Android NNAPI check ──
    if is_android() and not npus:
        try:
            from pathlib import Path
            # NNAPI is available on Android 8.1+ — check SDK level
            import subprocess as sp
            r = sp.run(["getprop", "ro.build.version.sdk"],
                       capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                sdk = int(r.stdout.strip())
                if sdk >= 27:  # Android 8.1
                    key = "android-nnapi"
                    if key not in seen:
                        # Try TFLite GPU delegate as backend
                        backend = "none"
                        try:
                            import tflite_runtime  # type: ignore[import-untyped]
                            backend = "tflite"
                        except ImportError:
                            try:
                                import tensorflow as tf  # type: ignore[import-untyped]
                                backend = "tflite"
                            except ImportError:
                                pass

                        cpu_name = _get_cpu_name()
                        tops = _estimate_tops(cpu_name)
                        vendor = _vendor_from_name(cpu_name)

                        npus.append(NPUDevice(
                            name=f"NNAPI ({cpu_name or 'Android'})",
                            vendor=vendor,
                            backend=backend,
                            tops=tops,
                        ))
                        seen.add(key)
        except Exception:
            pass

    return npus


def _detect_arm_npu_backend(soc) -> str:
    """Determine best available backend for an ARM NPU."""
    # QNN EP for Qualcomm Hexagon
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
        eps = ort.get_available_providers()
        if "QNNExecutionProvider" in eps:
            return "onnxrt-qnn"
        if "DmlExecutionProvider" in eps:
            return "onnxrt-dml"
    except ImportError:
        pass

    # TFLite
    try:
        import tflite_runtime  # type: ignore[import-untyped]
        return "tflite"
    except ImportError:
        pass
    try:
        import tensorflow  # type: ignore[import-untyped]
        return "tflite"
    except ImportError:
        pass

    return "none"


def _probe_windows_npu(seen: set[str]) -> List[NPUDevice]:
    """Detect NPU via Windows WMI / Device Manager."""
    npus: List[NPUDevice] = []
    try:
        # Check for NPU in PnP devices (class = NeuralProcessor or MachineLearning)
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-PnpDevice -Class 'MachineLearningModel','NeuralProcessor','AI','Processor' "
             "-ErrorAction SilentlyContinue | "
             "Where-Object { $_.FriendlyName -match 'NPU|Neural|AI' } | "
             "Select-Object -ExpandProperty FriendlyName"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                name = line.strip()
                if not name or name.lower() in seen:
                    continue
                vendor = _vendor_from_name(name)
                cpu_name = _get_cpu_name()
                tops = _estimate_tops(name) or _estimate_tops(cpu_name)
                npus.append(NPUDevice(
                    name=name, vendor=vendor, backend="none", tops=tops,
                ))
                seen.add(name.lower())
    except Exception:
        pass

    # Fallback: check CPU name for known NPU-equipped processors
    if not npus:
        cpu_name = _get_cpu_name()
        tops = _estimate_tops(cpu_name)
        if tops > 0:
            vendor = _vendor_from_name(cpu_name)
            name = f"{vendor} NPU (in {cpu_name})"
            if name.lower() not in seen:
                npus.append(NPUDevice(
                    name=name, vendor=vendor, backend="none", tops=tops,
                ))
                seen.add(name.lower())

    return npus


def _probe_linux_npu(seen: set[str]) -> List[NPUDevice]:
    """Detect NPU via Linux sysfs / accel subsystem."""
    npus: List[NPUDevice] = []
    try:
        import pathlib
        # Intel NPU driver exposes /dev/accel/accel*
        accel_path = pathlib.Path("/dev/accel")
        if accel_path.exists():
            devices = list(accel_path.glob("accel*"))
            if devices:
                cpu_name = _get_cpu_name()
                tops = _estimate_tops(cpu_name)
                vendor = _vendor_from_name(cpu_name)
                name = f"{vendor} NPU ({cpu_name})" if cpu_name else "Linux NPU"
                npus.append(NPUDevice(
                    name=name, vendor=vendor, backend="none", tops=tops,
                ))
                seen.add(name.lower())

        # XDNA driver for AMD
        xdna = pathlib.Path("/sys/class/accel")
        if xdna.exists() and not npus:
            for dev in xdna.iterdir():
                if (dev / "device" / "uevent").exists():
                    uevent = (dev / "device" / "uevent").read_text()
                    if "xdna" in uevent.lower() or "npu" in uevent.lower():
                        cpu_name = _get_cpu_name()
                        tops = _estimate_tops(cpu_name)
                        name = f"AMD XDNA NPU ({cpu_name})"
                        npus.append(NPUDevice(
                            name=name, vendor="AMD", backend="none", tops=tops,
                        ))
                        break
    except Exception:
        pass
    return npus


def _probe_macos_npu(seen: set[str]) -> List[NPUDevice]:
    """Detect Apple Neural Engine on macOS."""
    npus: List[NPUDevice] = []
    try:
        # Check for ANE via ioreg
        r = subprocess.run(
            ["ioreg", "-c", "AppleARMIODevice", "-r", "-d", "1"],
            capture_output=True, text=True, timeout=5,
        )
        has_ane = "ane" in r.stdout.lower() or "neural" in r.stdout.lower()

        if not has_ane:
            # All Apple Silicon has ANE
            r2 = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if r2.returncode == 0 and "apple" in r2.stdout.lower():
                has_ane = True

        if has_ane:
            cpu_name = _get_cpu_name()
            tops = _estimate_tops(cpu_name)
            name = f"Apple Neural Engine ({cpu_name})"
            vendor = "Apple"

            # Check if coremltools is available
            backend = "none"
            try:
                import coremltools  # type: ignore[import-untyped]
                backend = "coreml"
            except ImportError:
                pass

            npus.append(NPUDevice(
                name=name, vendor=vendor, backend=backend, tops=tops,
            ))
    except Exception:
        pass
    return npus


# ── Helpers ─────────────────────────────────────────────────────────────

_cpu_name_cache: Optional[str] = None


def _get_cpu_name() -> str:
    """Get the host CPU brand string (cached)."""
    global _cpu_name_cache
    if _cpu_name_cache is not None:
        return _cpu_name_cache

    name = ""
    try:
        from pyaccelerate.cpu import detect as cpu_detect
        name = cpu_detect().brand
    except Exception:
        pass

    if not name:
        name = platform.processor() or ""

    _cpu_name_cache = name
    return name


def _detect_onnxrt_vendor() -> str:
    """Guess vendor from CPU name for generic ONNX Runtime EPs."""
    cpu = _get_cpu_name()
    return _vendor_from_name(cpu) if cpu else "unknown"


# ═══════════════════════════════════════════════════════════════════════════
#  Public convenience API
# ═══════════════════════════════════════════════════════════════════════════

def npu_available() -> bool:
    """True if at least one usable NPU exists."""
    return any(n.usable for n in detect_all())


def best_npu() -> Optional[NPUDevice]:
    """Return the highest-scored usable NPU, or None."""
    detect_all()
    if _best_npu and _best_npu.usable:
        return _best_npu
    return None


def get_npu_info() -> Dict[str, str]:
    """Info dict for the best NPU (or fallback)."""
    npus = detect_all()
    top = npus[0] if npus else None
    if top is None or not top.usable:
        hw = top.name if top else "N/A"
        return {
            "available": "false",
            "backend": "none",
            "device": hw or "N/A",
            "note": "No NPU compute framework — install openvino/onnxruntime",
        }
    return {
        "available": "true",
        "backend": top.backend,
        "device": top.name,
        "vendor": top.vendor,
        "tops": f"{top.tops:.1f}",
        "score": str(top.score),
    }


def get_all_npus_info() -> List[Dict[str, str]]:
    """Info dicts for all detected NPUs (best-first)."""
    return [n.as_dict() for n in detect_all()]


def get_install_hint() -> str:
    """Suggest pip install commands for NPU frameworks."""
    npus = detect_all()
    usable = [n for n in npus if n.usable]
    if usable:
        return ""
    if not npus:
        return ""

    hints: list[str] = []
    for n in npus:
        vl = n.vendor.lower()
        if "intel" in vl:
            hints.append("pip install openvino")
            hints.append("pip install onnxruntime-openvino")
        elif "amd" in vl:
            hints.append("pip install onnxruntime-directml")
        elif "qualcomm" in vl:
            hints.append("pip install onnxruntime-qnn")
        elif "apple" in vl:
            hints.append("pip install coremltools")
        elif "mediatek" in vl or "samsung" in vl or "google" in vl or "hisilicon" in vl:
            hints.append("pip install tflite-runtime  # Android NNAPI")
        else:
            hints.append("pip install onnxruntime-directml")

    if hints:
        return "Install NPU support: " + " or ".join(sorted(set(hints)))
    return ""
