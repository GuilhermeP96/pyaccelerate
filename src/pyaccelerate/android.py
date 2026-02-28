"""
pyaccelerate.android — Android & Termux platform detection and helpers.

Detects whether we are running on Android (via Termux, Pydroid, QPython,
or native CPython) and exposes device-level information:

  - SoC / chipset identification (Snapdragon, Exynos, Dimensity, Tensor)
  - Thermal throttling state
  - Battery-aware scheduling hints
  - Available kernel interfaces (/sys/class/kgsl, /dev/ion, etc.)

Thread-safe. All results are cached after first detection.

Usage::

    from pyaccelerate.android import is_android, get_device_info

    if is_android():
        info = get_device_info()
        print(info["soc"], info["board"])
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("pyaccelerate.android")


# ═══════════════════════════════════════════════════════════════════════════
#  Android / Termux Detection
# ═══════════════════════════════════════════════════════════════════════════

_is_android: Optional[bool] = None
_is_termux: Optional[bool] = None
_device_info: Optional[Dict[str, str]] = None


def is_android() -> bool:
    """Return True if running on Android (Termux, Pydroid, etc.)."""
    global _is_android
    if _is_android is not None:
        return _is_android

    checks = [
        # Termux sets this
        "ANDROID_ROOT" in os.environ,
        "ANDROID_DATA" in os.environ,
        # Termux prefix
        os.path.isdir("/data/data/com.termux"),
        # Android-specific paths
        os.path.isfile("/system/build.prop"),
        # Platform hints
        "android" in platform.platform().lower(),
        "aarch64" in platform.machine().lower() and os.path.isdir("/system"),
    ]
    _is_android = any(checks)

    if _is_android:
        log.info("Android platform detected")
    return _is_android


def is_termux() -> bool:
    """Return True if running inside Termux."""
    global _is_termux
    if _is_termux is not None:
        return _is_termux

    _is_termux = (
        "TERMUX_VERSION" in os.environ
        or "PREFIX" in os.environ and "/com.termux" in os.environ.get("PREFIX", "")
        or os.path.isdir("/data/data/com.termux/files")
    )
    return _is_termux


def is_arm() -> bool:
    """Return True if running on ARM architecture (ARM32 or ARM64/aarch64)."""
    machine = platform.machine().lower()
    return any(arm in machine for arm in ("aarch64", "arm64", "armv7", "armv8", "arm"))


# ═══════════════════════════════════════════════════════════════════════════
#  SoC / Chipset Database
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SoCInfo:
    """System-on-Chip information."""
    name: str = ""                 # e.g. "Snapdragon 8 Gen 3"
    vendor: str = ""               # "Qualcomm", "Samsung", "MediaTek", "Google", "HiSilicon"
    cpu_arch: str = ""             # "ARMv9", "ARMv8.2-A", etc.
    cpu_cores_big: int = 0         # Performance cores
    cpu_cores_mid: int = 0         # Efficiency+ cores (tri-cluster)
    cpu_cores_little: int = 0      # Efficiency cores
    gpu_name: str = ""             # "Adreno 750", "Mali-G720"
    gpu_cores: int = 0
    npu_name: str = ""             # "Hexagon DSP", "Samsung NPU"
    npu_tops: float = 0.0
    process_nm: int = 0            # Manufacturing process (nm)
    board: str = ""                # getprop ro.product.board


# Known SoC database (keyed by board/hardware/chipset patterns)
_SOC_DATABASE: Dict[str, SoCInfo] = {
    # ── Qualcomm Snapdragon ──
    "sm8750": SoCInfo("Snapdragon 8 Elite", "Qualcomm", "ARMv9.2", 2, 6, 0,
                       "Adreno 830", 0, "Hexagon NPU", 80.0, 3),
    "sm8650": SoCInfo("Snapdragon 8 Gen 3", "Qualcomm", "ARMv9.2", 1, 3, 4,
                       "Adreno 750", 0, "Hexagon NPU", 73.0, 4),
    "sm8550": SoCInfo("Snapdragon 8 Gen 2", "Qualcomm", "ARMv9", 1, 2, 4,
                       "Adreno 740", 0, "Hexagon NPU", 36.0, 4),
    "sm8450": SoCInfo("Snapdragon 8 Gen 1", "Qualcomm", "ARMv9", 1, 3, 4,
                       "Adreno 730", 0, "Hexagon NPU", 27.0, 4),
    "sm8350": SoCInfo("Snapdragon 888", "Qualcomm", "ARMv8.4-A", 1, 3, 4,
                       "Adreno 660", 0, "Hexagon 780", 26.0, 5),
    "sm8250": SoCInfo("Snapdragon 865", "Qualcomm", "ARMv8.2-A", 1, 3, 4,
                       "Adreno 650", 0, "Hexagon 698", 15.0, 7),
    "sm7675": SoCInfo("Snapdragon 7+ Gen 3", "Qualcomm", "ARMv9", 1, 3, 4,
                       "Adreno 732", 0, "Hexagon NPU", 45.0, 4),
    "sm7550": SoCInfo("Snapdragon 7 Gen 3", "Qualcomm", "ARMv9", 1, 3, 4,
                       "Adreno 720", 0, "Hexagon NPU", 20.0, 4),
    "sm6450": SoCInfo("Snapdragon 6 Gen 1", "Qualcomm", "ARMv9", 0, 4, 4,
                       "Adreno 710", 0, "Hexagon NPU", 12.0, 4),
    # Snapdragon laptops
    "sc8380xp": SoCInfo("Snapdragon X Elite", "Qualcomm", "ARMv8.7-A", 12, 0, 0,
                         "Adreno X1", 0, "Hexagon NPU", 45.0, 4),
    "sc8280xp": SoCInfo("Snapdragon 8cx Gen 3", "Qualcomm", "ARMv8.4-A", 4, 4, 0,
                         "Adreno 690", 0, "Hexagon 780", 29.0, 5),

    # ── Samsung Exynos ──
    "s5e9945": SoCInfo("Exynos 2500", "Samsung", "ARMv9.2", 1, 3, 4,
                        "Xclipse 950", 0, "Samsung NPU", 50.0, 3),
    "s5e9925": SoCInfo("Exynos 2200", "Samsung", "ARMv9", 1, 3, 4,
                        "Xclipse 920", 0, "Samsung NPU", 34.7, 4),
    "s5e8835": SoCInfo("Exynos 1380", "Samsung", "ARMv8.2-A", 0, 4, 4,
                        "Mali-G68 MP5", 5, "Samsung NPU", 5.2, 5),
    "exynos990": SoCInfo("Exynos 990", "Samsung", "ARMv8.2-A", 2, 2, 4,
                          "Mali-G77 MP11", 11, "Samsung NPU", 10.0, 7),
    "exynos2100": SoCInfo("Exynos 2100", "Samsung", "ARMv8.2-A", 1, 3, 4,
                           "Mali-G78 MP14", 14, "Samsung NPU", 26.0, 5),

    # ── Google Tensor ──
    "gs101": SoCInfo("Tensor G1", "Google", "ARMv8.2-A", 2, 2, 4,
                      "Mali-G78 MP20", 20, "Google TPU", 18.0, 5),
    "gs201": SoCInfo("Tensor G2", "Google", "ARMv9", 2, 2, 4,
                      "Mali-G710 MP7", 7, "Google TPU v2", 22.0, 4),
    "zuma": SoCInfo("Tensor G3", "Google", "ARMv9", 1, 4, 4,
                     "Immortalis-G715 MC10", 10, "Google TPU v3", 30.0, 4),
    "zumapro": SoCInfo("Tensor G4", "Google", "ARMv9.2", 1, 3, 4,
                        "Mali-G715 MC7", 7, "Google TPU v4", 35.0, 4),

    # ── MediaTek Dimensity ──
    "mt6989": SoCInfo("Dimensity 9300", "MediaTek", "ARMv9.2", 4, 4, 0,
                       "Immortalis-G720 MC12", 12, "MediaTek APU 790", 46.0, 4),
    "mt6985": SoCInfo("Dimensity 9200", "MediaTek", "ARMv9", 1, 3, 4,
                       "Immortalis-G715 MC11", 11, "MediaTek APU 690", 35.0, 4),
    "mt6983": SoCInfo("Dimensity 9000", "MediaTek", "ARMv9", 1, 3, 4,
                       "Mali-G710 MC10", 10, "MediaTek APU 590", 25.0, 4),
    "mt6893": SoCInfo("Dimensity 1200", "MediaTek", "ARMv8.2-A", 1, 3, 4,
                       "Mali-G77 MC9", 9, "MediaTek APU 3.0", 9.0, 6),
    "mt6891": SoCInfo("Dimensity 1100", "MediaTek", "ARMv8.2-A", 0, 4, 4,
                       "Mali-G77 MC9", 9, "MediaTek APU 3.0", 7.0, 6),
    "mt6877": SoCInfo("Dimensity 900", "MediaTek", "ARMv8.2-A", 0, 2, 6,
                       "Mali-G68 MC4", 4, "MediaTek APU 3.0", 5.0, 6),
    "mt6897": SoCInfo("Dimensity 8300", "MediaTek", "ARMv9.2", 0, 4, 4,
                       "Mali-G615 MC6", 6, "MediaTek APU 780", 20.0, 4),

    # ── HiSilicon Kirin ──
    "kirin9000": SoCInfo("Kirin 9000", "HiSilicon", "ARMv8.2-A", 1, 3, 4,
                          "Mali-G78 MP24", 24, "Da Vinci NPU", 12.0, 5),
    "kirin9010": SoCInfo("Kirin 9010", "HiSilicon", "ARMv9", 1, 3, 4,
                          "Maleoon 910", 0, "Da Vinci NPU", 25.0, 7),

    # ── Unisoc (budget) ──
    "ums9230": SoCInfo("Unisoc T616", "Unisoc", "ARMv8.2-A", 0, 2, 6,
                        "Mali-G57 MP1", 1, "N/A", 0.0, 12),
}


def _lookup_soc(board: str, hardware: str, chipset: str) -> Optional[SoCInfo]:
    """Try to match board/hardware/chipset against known SoC database.

    Uses exact key match first, then substring match (key in candidate).
    """
    candidates = [board.lower().strip(), hardware.lower().strip(), chipset.lower().strip()]
    candidates = [c for c in candidates if c]

    # Pass 1: exact match — candidate equals a key
    for key, soc in _SOC_DATABASE.items():
        for candidate in candidates:
            if key == candidate:
                soc_copy = SoCInfo(**{k: getattr(soc, k) for k in soc.__dataclass_fields__})
                soc_copy.board = board
                return soc_copy

    # Pass 2: key is a substring of candidate (e.g. "sm8650" in "qcom,sm8650")
    for key, soc in _SOC_DATABASE.items():
        for candidate in candidates:
            if key in candidate:
                soc_copy = SoCInfo(**{k: getattr(soc, k) for k in soc.__dataclass_fields__})
                soc_copy.board = board
                return soc_copy

    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Device Info
# ═══════════════════════════════════════════════════════════════════════════

def get_device_info() -> Dict[str, str]:
    """Gather Android device information.

    Uses ``getprop`` (Android) or falls back to platform info.
    Returns a dict with keys: model, manufacturer, board, hardware,
    chipset, soc, android_version, sdk_level, abi.
    """
    global _device_info
    if _device_info is not None:
        return _device_info

    info: Dict[str, str] = {}

    if is_android():
        info = _gather_android_props()
    elif is_arm():
        info = _gather_arm_linux_info()
    else:
        info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

    _device_info = info
    return info


def _getprop(key: str) -> str:
    """Read a single Android system property."""
    try:
        r = subprocess.run(
            ["getprop", key], capture_output=True, text=True, timeout=3,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _gather_android_props() -> Dict[str, str]:
    """Gather device info from Android getprop."""
    props: Dict[str, str] = {}

    prop_keys = {
        "model": "ro.product.model",
        "manufacturer": "ro.product.manufacturer",
        "board": "ro.product.board",
        "hardware": "ro.hardware",
        "chipset": "ro.hardware.chipname",
        "android_version": "ro.build.version.release",
        "sdk_level": "ro.build.version.sdk",
        "abi": "ro.product.cpu.abi",
        "soc_manufacturer": "ro.soc.manufacturer",
        "soc_model": "ro.soc.model",
    }

    for dest_key, prop_key in prop_keys.items():
        val = _getprop(prop_key)
        if val:
            props[dest_key] = val

    # Build SoC name from available info
    soc_parts = []
    if props.get("soc_manufacturer"):
        soc_parts.append(props["soc_manufacturer"])
    if props.get("soc_model"):
        soc_parts.append(props["soc_model"])
    if soc_parts:
        props["soc"] = " ".join(soc_parts)
    elif props.get("chipset"):
        props["soc"] = props["chipset"]
    elif props.get("hardware"):
        props["soc"] = props["hardware"]

    props["platform"] = "Android"
    props["machine"] = platform.machine()

    return props


def _gather_arm_linux_info() -> Dict[str, str]:
    """Gather ARM info on non-Android Linux (e.g. Raspberry Pi, ARM server)."""
    info: Dict[str, str] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # Device tree model
    try:
        model = Path("/proc/device-tree/model").read_text().strip().rstrip("\x00")
        info["model"] = model
    except Exception:
        pass

    # /proc/cpuinfo Hardware field (common on ARM)
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo.splitlines():
            low = line.lower()
            if low.startswith("hardware"):
                info["hardware"] = line.split(":", 1)[-1].strip()
            elif low.startswith("cpu implementer"):
                info["cpu_implementer"] = line.split(":", 1)[-1].strip()
            elif low.startswith("cpu part"):
                info["cpu_part"] = line.split(":", 1)[-1].strip()
            elif low.startswith("features") or low.startswith("flags"):
                info["cpu_features"] = line.split(":", 1)[-1].strip()
    except Exception:
        pass

    return info


def get_soc_info() -> Optional[SoCInfo]:
    """Identify the SoC from device properties."""
    info = get_device_info()
    board = info.get("board", "")
    hardware = info.get("hardware", "")
    chipset = info.get("chipset", info.get("soc", ""))
    return _lookup_soc(board, hardware, chipset)


# ═══════════════════════════════════════════════════════════════════════════
#  Thermal & Battery
# ═══════════════════════════════════════════════════════════════════════════

def get_thermal_zones() -> Dict[str, float]:
    """Read thermal zone temperatures (Linux/Android)."""
    temps: Dict[str, float] = {}
    try:
        thermal_dir = Path("/sys/class/thermal")
        if thermal_dir.exists():
            for zone in thermal_dir.glob("thermal_zone*"):
                try:
                    name = (zone / "type").read_text().strip()
                    temp_raw = int((zone / "temp").read_text().strip())
                    temps[name] = temp_raw / 1000.0  # millidegrees → °C
                except Exception:
                    continue
    except Exception:
        pass
    return temps


def get_battery_info() -> Dict[str, Any]:
    """Read battery state (Android / Linux with power_supply)."""
    info: Dict[str, Any] = {}
    try:
        batt_paths = [
            Path("/sys/class/power_supply/battery"),
            Path("/sys/class/power_supply/BAT0"),
            Path("/sys/class/power_supply/BAT1"),
        ]
        for batt in batt_paths:
            if not batt.exists():
                continue
            for field in ("capacity", "status", "current_now", "voltage_now",
                          "temperature", "health", "technology"):
                f = batt / field
                if f.exists():
                    val = f.read_text().strip()
                    try:
                        info[field] = int(val)
                    except ValueError:
                        info[field] = val
            if info:
                break
    except Exception:
        pass
    return info


def is_thermally_throttled() -> bool:
    """Check if the device appears to be thermally throttled."""
    temps = get_thermal_zones()
    for name, temp in temps.items():
        nl = name.lower()
        if any(k in nl for k in ("cpu", "soc", "skin")) and temp > 80.0:
            return True
    return False


def get_cpu_freq_info() -> Dict[str, Dict[str, float]]:
    """Read per-CPU frequency info from sysfs (ARM/Linux)."""
    result: Dict[str, Dict[str, float]] = {}
    try:
        cpu_dir = Path("/sys/devices/system/cpu")
        for cpu in sorted(cpu_dir.glob("cpu[0-9]*")):
            cpufreq = cpu / "cpufreq"
            if not cpufreq.exists():
                continue
            freqs: Dict[str, float] = {}
            for param in ("scaling_cur_freq", "scaling_min_freq", "scaling_max_freq",
                          "cpuinfo_min_freq", "cpuinfo_max_freq"):
                f = cpufreq / param
                if f.exists():
                    try:
                        freqs[param] = int(f.read_text().strip()) / 1000.0  # kHz → MHz
                    except Exception:
                        pass
            if freqs:
                result[cpu.name] = freqs
    except Exception:
        pass
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  ARM Core Identification
# ═══════════════════════════════════════════════════════════════════════════

# ARM CPU part numbers → core names
_ARM_PART_NAMES: Dict[str, str] = {
    "0xd03": "Cortex-A53",
    "0xd04": "Cortex-A35",
    "0xd05": "Cortex-A55",
    "0xd06": "Cortex-A65",
    "0xd07": "Cortex-A57",
    "0xd08": "Cortex-A72",
    "0xd09": "Cortex-A73",
    "0xd0a": "Cortex-A75",
    "0xd0b": "Cortex-A76",
    "0xd0c": "Neoverse N1",
    "0xd0d": "Cortex-A77",
    "0xd0e": "Cortex-A76AE",
    "0xd40": "Neoverse V1",
    "0xd41": "Cortex-A78",
    "0xd42": "Cortex-A78AE",
    "0xd43": "Cortex-A65AE",
    "0xd44": "Cortex-X1",
    "0xd46": "Cortex-A510",
    "0xd47": "Cortex-A710",
    "0xd48": "Cortex-X2",
    "0xd49": "Neoverse N2",
    "0xd4a": "Neoverse E1",
    "0xd4b": "Cortex-A78C",
    "0xd4d": "Cortex-A715",
    "0xd4e": "Cortex-X3",
    "0xd4f": "Neoverse V2",
    "0xd80": "Cortex-A520",
    "0xd81": "Cortex-A720",
    "0xd82": "Cortex-X4",
    "0xd83": "Neoverse V3",
    "0xd84": "Neoverse N3",
    "0xd85": "Cortex-X925",
    "0xd87": "Cortex-A725",
}

# ARM implementer IDs
_ARM_IMPLEMENTERS: Dict[str, str] = {
    "0x41": "ARM",
    "0x42": "Broadcom",
    "0x43": "Cavium",
    "0x44": "DEC",
    "0x46": "Fujitsu",
    "0x48": "HiSilicon",
    "0x4e": "NVIDIA",
    "0x50": "APM",
    "0x51": "Qualcomm",
    "0x53": "Samsung",
    "0x56": "Marvell",
    "0x61": "Apple",
    "0x63": "Intel",   # historical
    "0x69": "Intel",
    "0xc0": "Ampere",
}


def detect_arm_cores() -> List[Dict[str, str]]:
    """Parse /proc/cpuinfo to identify individual ARM CPU cores.

    Returns list of dicts with keys: processor, implementer, part, name, variant.
    On big.LITTLE SoCs you'll see different parts for big/LITTLE clusters.
    """
    cores: List[Dict[str, str]] = []
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        current: Dict[str, str] = {}

        for line in cpuinfo.splitlines():
            line = line.strip()
            if not line:
                if current:
                    cores.append(current)
                    current = {}
                continue

            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip()

            if key == "processor":
                current["processor"] = val
            elif key == "cpu implementer":
                current["implementer"] = val
                impl_name = _ARM_IMPLEMENTERS.get(val, "Unknown")
                current["implementer_name"] = impl_name
            elif key == "cpu part":
                current["part"] = val
                part_name = _ARM_PART_NAMES.get(val, f"Unknown ({val})")
                current["name"] = part_name
            elif key == "cpu variant":
                current["variant"] = val
            elif key == "cpu revision":
                current["revision"] = val

        if current:
            cores.append(current)

    except Exception:
        pass
    return cores


def detect_big_little() -> Dict[str, List[int]]:
    """Detect big.LITTLE core clusters from /proc/cpuinfo.

    Returns a dict mapping core type name to list of CPU indices.
    E.g.: {"Cortex-A78": [0,1,2,3], "Cortex-A55": [4,5,6,7]}
    """
    cores = detect_arm_cores()
    clusters: Dict[str, List[int]] = {}

    for core in cores:
        name = core.get("name", "Unknown")
        proc = core.get("processor", "")
        try:
            idx = int(proc)
        except ValueError:
            continue
        clusters.setdefault(name, []).append(idx)

    return clusters


def get_arm_features() -> List[str]:
    """Get ARM CPU feature flags from /proc/cpuinfo.

    Common ARM features: neon, asimd, fp, fphp, asimdhp, sve, sve2,
    aes, sha1, sha2, crc32, atomics, i8mm, bf16, etc.
    """
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo.splitlines():
            low = line.lower().strip()
            if low.startswith("features"):
                return sorted(line.split(":", 1)[-1].strip().split())
    except Exception:
        pass
    return []
