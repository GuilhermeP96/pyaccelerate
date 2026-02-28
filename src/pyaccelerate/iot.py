"""
pyaccelerate.iot — IoT & Single-Board Computer (SBC) platform detection.

Detects Raspberry Pi, NVIDIA Jetson, BeagleBone, Orange Pi, Pine64, and
other embedded Linux boards plus MicroPython / CircuitPython runtimes.

Provides:
  - ``SBCInfo`` dataclass with board model, SoC, RAM, GPU, NPU, peripherals
  - SBC SoC database (BCM27xx, Tegra, Allwinner, Rockchip, TI AM3xxx, NXP i.MX)
  - MicroPython / CircuitPython detection
  - Thermal & fan monitoring for fanless boards
  - ``recommend_iot_workers()`` — memory-constrained pool sizing
  - Helper for NVIDIA Jetson power models (``nvpmodel``)

Thread-safe.  All results are cached after first detection.

Usage::

    from pyaccelerate.iot import is_sbc, detect_sbc, recommend_iot_workers

    if is_sbc():
        board = detect_sbc()
        print(board.board_name, board.soc_name, f"{board.ram_mb} MB")
        print("Workers:", recommend_iot_workers())
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("pyaccelerate.iot")


# ═══════════════════════════════════════════════════════════════════════════
#  Platform Detection
# ═══════════════════════════════════════════════════════════════════════════

_is_sbc: Optional[bool] = None
_is_micropython: Optional[bool] = None
_is_circuitpython: Optional[bool] = None
_sbc_info: Optional["SBCInfo"] = None


def is_micropython() -> bool:
    """Return True if running under MicroPython."""
    global _is_micropython
    if _is_micropython is not None:
        return _is_micropython
    try:
        import sys
        _is_micropython = hasattr(sys, "implementation") and sys.implementation.name == "micropython"
    except Exception:
        _is_micropython = False
    return _is_micropython


def is_circuitpython() -> bool:
    """Return True if running under CircuitPython."""
    global _is_circuitpython
    if _is_circuitpython is not None:
        return _is_circuitpython
    try:
        import sys
        _is_circuitpython = hasattr(sys, "implementation") and sys.implementation.name == "circuitpython"
    except Exception:
        _is_circuitpython = False
    return _is_circuitpython


def is_sbc() -> bool:
    """Return True if running on a known Single-Board Computer.

    Checks device-tree model, compatible strings, and known paths such as
    ``/proc/device-tree/model``, ``/etc/nv_tegra_release``, etc.
    """
    global _is_sbc
    if _is_sbc is not None:
        return _is_sbc

    # MicroPython / CircuitPython boards are always SBCs
    if is_micropython() or is_circuitpython():
        _is_sbc = True
        return True

    model = _read_dt_model()
    compatible = _read_dt_compatible()

    checks = [
        # Raspberry Pi
        "raspberry pi" in model.lower(),
        # NVIDIA Jetson
        "jetson" in model.lower(),
        os.path.isfile("/etc/nv_tegra_release"),
        "nvidia" in compatible.lower() and "tegra" in compatible.lower(),
        # BeagleBone
        "beaglebone" in model.lower(),
        "beagleboard" in model.lower(),
        "ti,am33" in compatible.lower(),
        "ti,am57" in compatible.lower(),
        # Orange Pi
        "orange pi" in model.lower(),
        # Pine64
        "pine64" in model.lower(),
        "pinebook" in model.lower(),
        # Banana Pi
        "banana pi" in model.lower(),
        "bananapi" in model.lower(),
        # Odroid
        "odroid" in model.lower(),
        # NXP i.MX
        "fsl,imx" in compatible.lower(),
        # Allwinner boards (generic)
        "allwinner" in compatible.lower(),
        # Rockchip boards (generic)
        "rockchip" in compatible.lower(),
        # Amlogic boards (generic)
        "amlogic" in compatible.lower(),
        # Google Coral dev board
        "coral" in model.lower(),
    ]

    _is_sbc = any(checks)
    if _is_sbc:
        log.info("SBC platform detected: %s", model or compatible[:80])
    return _is_sbc


def is_jetson() -> bool:
    """Return True if running on an NVIDIA Jetson platform."""
    model = _read_dt_model().lower()
    compatible = _read_dt_compatible().lower()
    return (
        "jetson" in model
        or os.path.isfile("/etc/nv_tegra_release")
        or ("nvidia" in compatible and "tegra" in compatible)
    )


def is_raspberry_pi() -> bool:
    """Return True if running on a Raspberry Pi."""
    model = _read_dt_model().lower()
    return "raspberry pi" in model


# ═══════════════════════════════════════════════════════════════════════════
#  SBC Info Dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SBCInfo:
    """Single-Board Computer information."""
    board_name: str = ""         # e.g. "Raspberry Pi 4 Model B"
    soc_name: str = ""           # e.g. "BCM2711"
    soc_vendor: str = ""         # "Broadcom", "NVIDIA", "Allwinner", etc.
    cpu_arch: str = ""           # "ARMv8-A", "ARMv7-A", "RISC-V"
    cpu_cores: int = 0
    cpu_max_mhz: float = 0.0
    ram_mb: int = 0              # Total RAM in MB
    gpu_name: str = ""           # "VideoCore VI", "Tegra Xavier GPU"
    gpu_cuda_cores: int = 0      # Jetson CUDA cores, 0 for others
    npu_name: str = ""           # "Jetson DLA", "Coral Edge TPU", etc.
    npu_tops: float = 0.0
    has_gpio: bool = False
    gpio_pins: int = 0           # e.g. 40 for RPi
    has_camera_csi: bool = False
    has_display_dsi: bool = False
    has_pcie: bool = False
    has_wifi: bool = False
    has_bluetooth: bool = False
    has_ethernet: bool = False
    usb_ports: int = 0
    storage_type: str = ""       # "microSD", "eMMC", "NVMe", etc.
    has_fan: bool = False        # Physical fan detected
    process_nm: int = 0          # SoC manufacturing process
    family: str = ""             # "raspberry_pi", "jetson", "beaglebone", etc.
    # Jetson-specific
    jetson_power_mode: str = ""  # e.g. "MAXN", "15W", "30W"
    jetson_l4t_version: str = "" # e.g. "35.4.1"


# ═══════════════════════════════════════════════════════════════════════════
#  SBC / SoC Database
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _SBCSoC:
    """Internal SoC record for lookup."""
    soc_name: str
    vendor: str
    cpu_arch: str
    cpu_cores: int
    cpu_max_mhz: float
    gpu_name: str
    gpu_cuda_cores: int = 0
    npu_name: str = ""
    npu_tops: float = 0.0
    process_nm: int = 0


# Keyed by patterns found in device-tree model or compatible strings
_SBC_SOC_DATABASE: Dict[str, _SBCSoC] = {
    # ── Raspberry Pi ──
    "bcm2712": _SBCSoC("BCM2712", "Broadcom", "ARMv8.2-A", 4, 2400.0,
                        "VideoCore VII", 0, "", 0.0, 16),
    "bcm2711": _SBCSoC("BCM2711", "Broadcom", "ARMv8-A", 4, 1800.0,
                        "VideoCore VI", 0, "", 0.0, 28),
    "bcm2837": _SBCSoC("BCM2837", "Broadcom", "ARMv8-A", 4, 1400.0,
                        "VideoCore IV", 0, "", 0.0, 40),
    "bcm2836": _SBCSoC("BCM2836", "Broadcom", "ARMv7-A", 4, 900.0,
                        "VideoCore IV", 0, "", 0.0, 40),
    "bcm2835": _SBCSoC("BCM2835", "Broadcom", "ARMv6", 1, 700.0,
                        "VideoCore IV", 0, "", 0.0, 40),

    # ── NVIDIA Jetson ──
    "jetson orin nano": _SBCSoC("Tegra Orin (Nano)", "NVIDIA", "ARMv8.2-A", 6, 1500.0,
                                 "Ampere (1024 CUDA)", 1024, "Jetson DLA", 40.0, 8),
    "jetson orin nx": _SBCSoC("Tegra Orin (NX)", "NVIDIA", "ARMv8.2-A", 8, 2000.0,
                               "Ampere (1024 CUDA)", 1024, "Jetson DLA", 100.0, 8),
    "jetson agx orin": _SBCSoC("Tegra Orin (AGX)", "NVIDIA", "ARMv8.2-A", 12, 2200.0,
                                "Ampere (2048 CUDA)", 2048, "Jetson DLA ×2", 275.0, 8),
    "jetson xavier nx": _SBCSoC("Tegra Xavier (NX)", "NVIDIA", "ARMv8.2-A", 6, 1900.0,
                                  "Volta (384 CUDA)", 384, "Jetson DLA ×2", 21.0, 12),
    "jetson agx xavier": _SBCSoC("Tegra Xavier (AGX)", "NVIDIA", "ARMv8.2-A", 8, 2265.0,
                                   "Volta (512 CUDA)", 512, "Jetson DLA ×2", 32.0, 12),
    "jetson tx2": _SBCSoC("Tegra TX2", "NVIDIA", "ARMv8-A", 6, 2000.0,
                           "Pascal (256 CUDA)", 256, "", 0.0, 16),
    "jetson nano": _SBCSoC("Tegra X1 (Nano)", "NVIDIA", "ARMv8-A", 4, 1430.0,
                            "Maxwell (128 CUDA)", 128, "", 0.0, 20),

    # ── BeagleBone ──
    "am3358": _SBCSoC("AM3358", "TI", "ARMv7-A", 1, 1000.0,
                       "PowerVR SGX530", 0, "", 0.0, 45),
    "am3359": _SBCSoC("AM3359", "TI", "ARMv7-A", 1, 1000.0,
                       "PowerVR SGX530", 0, "", 0.0, 45),
    "am5729": _SBCSoC("AM5729", "TI", "ARMv7-A", 2, 1500.0,
                       "PowerVR SGX544", 0, "", 0.0, 28),

    # ── Allwinner (Orange Pi, Pine64, Banana Pi, etc.) ──
    "h616": _SBCSoC("H616", "Allwinner", "ARMv8-A", 4, 1512.0,
                     "Mali-G31 MP2", 0, "", 0.0, 28),
    "h618": _SBCSoC("H618", "Allwinner", "ARMv8-A", 4, 1512.0,
                     "Mali-G31 MP2", 0, "", 0.0, 28),
    "h6": _SBCSoC("H6", "Allwinner", "ARMv8-A", 4, 1800.0,
                   "Mali-T720 MP2", 0, "", 0.0, 28),
    "h5": _SBCSoC("H5", "Allwinner", "ARMv8-A", 4, 1296.0,
                   "Mali-450 MP4", 0, "", 0.0, 40),
    "h3": _SBCSoC("H3", "Allwinner", "ARMv7-A", 4, 1296.0,
                   "Mali-400 MP2", 0, "", 0.0, 40),
    "a64": _SBCSoC("A64", "Allwinner", "ARMv8-A", 4, 1152.0,
                    "Mali-400 MP2", 0, "", 0.0, 40),

    # ── Rockchip (Orange Pi 5, Pine64, Radxa, etc.) ──
    "rk3588": _SBCSoC("RK3588", "Rockchip", "ARMv8.2-A", 8, 2400.0,
                       "Mali-G610 MP4", 0, "Rockchip NPU", 6.0, 8),
    "rk3588s": _SBCSoC("RK3588S", "Rockchip", "ARMv8.2-A", 8, 2400.0,
                        "Mali-G610 MP4", 0, "Rockchip NPU", 6.0, 8),
    "rk3399": _SBCSoC("RK3399", "Rockchip", "ARMv8-A", 6, 2000.0,
                       "Mali-T860 MP4", 0, "", 0.0, 14),
    "rk3328": _SBCSoC("RK3328", "Rockchip", "ARMv8-A", 4, 1512.0,
                       "Mali-450 MP2", 0, "", 0.0, 28),
    "rk3566": _SBCSoC("RK3566", "Rockchip", "ARMv8.2-A", 4, 1800.0,
                       "Mali-G52 2EE", 0, "Rockchip NPU", 0.8, 22),
    "rk3568": _SBCSoC("RK3568", "Rockchip", "ARMv8.2-A", 4, 2000.0,
                       "Mali-G52 2EE", 0, "Rockchip NPU", 1.0, 22),

    # ── NXP i.MX ──
    "imx8mq": _SBCSoC("i.MX 8M Quad", "NXP", "ARMv8-A", 4, 1500.0,
                       "Vivante GC7000L", 0, "", 0.0, 14),
    "imx8mm": _SBCSoC("i.MX 8M Mini", "NXP", "ARMv8-A", 4, 1800.0,
                       "Vivante GC NanoUltra", 0, "", 0.0, 14),
    "imx8mp": _SBCSoC("i.MX 8M Plus", "NXP", "ARMv8-A", 4, 1800.0,
                       "Vivante GC7000UL", 0, "Ethos NPU", 2.3, 14),

    # ── Amlogic (Odroid, Libre, etc.) ──
    "s905x3": _SBCSoC("S905X3", "Amlogic", "ARMv8-A", 4, 2100.0,
                       "Mali-G31 MP2", 0, "", 0.0, 12),
    "s922x": _SBCSoC("S922X", "Amlogic", "ARMv8.2-A", 6, 2200.0,
                      "Mali-G52 MP6", 0, "", 0.0, 12),
    "a311d": _SBCSoC("A311D", "Amlogic", "ARMv8.2-A", 6, 2200.0,
                      "Mali-G52 MP4", 0, "Amlogic NPU", 5.0, 12),

    # ── Google Coral ──
    "coral": _SBCSoC("i.MX 8M (Coral)", "NXP", "ARMv8-A", 4, 1500.0,
                      "Vivante GC7000L", 0, "Google Edge TPU", 4.0, 14),
}


# Known board-name → family mapping
_BOARD_FAMILIES: Dict[str, str] = {
    "raspberry pi": "raspberry_pi",
    "jetson": "jetson",
    "beaglebone": "beaglebone",
    "beagleboard": "beaglebone",
    "orange pi": "orange_pi",
    "pine64": "pine64",
    "pinebook": "pine64",
    "banana pi": "banana_pi",
    "bananapi": "banana_pi",
    "odroid": "odroid",
    "coral": "coral",
    "radxa": "radxa",
    "rock pi": "radxa",
    "libre": "libre",
    "nanopi": "nanopi",
    "khadas": "khadas",
}


# ═══════════════════════════════════════════════════════════════════════════
#  Device-tree Helpers
# ═══════════════════════════════════════════════════════════════════════════

_dt_model_cache: Optional[str] = None
_dt_compat_cache: Optional[str] = None


def _read_dt_model() -> str:
    """Read device-tree model (e.g. 'Raspberry Pi 4 Model B Rev 1.4')."""
    global _dt_model_cache
    if _dt_model_cache is not None:
        return _dt_model_cache
    _dt_model_cache = ""
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            _dt_model_cache = Path(p).read_text().strip().rstrip("\x00")
            break
        except Exception:
            pass
    return _dt_model_cache


def _read_dt_compatible() -> str:
    """Read device-tree compatible strings (null-separated → space-joined)."""
    global _dt_compat_cache
    if _dt_compat_cache is not None:
        return _dt_compat_cache
    _dt_compat_cache = ""
    try:
        raw = Path("/proc/device-tree/compatible").read_bytes()
        _dt_compat_cache = raw.replace(b"\x00", b" ").decode(errors="replace").strip()
    except Exception:
        pass
    return _dt_compat_cache


# ═══════════════════════════════════════════════════════════════════════════
#  SBC Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_sbc() -> Optional[SBCInfo]:
    """Detect and return SBC information.  Returns None if not an SBC.  Cached."""
    global _sbc_info
    if _sbc_info is not None:
        return _sbc_info

    if not is_sbc():
        return None

    model = _read_dt_model()
    compatible = _read_dt_compatible()

    info = SBCInfo()
    info.board_name = model or "Unknown SBC"

    # ── Lookup SoC in database ──
    soc = _lookup_sbc_soc(model, compatible)
    if soc:
        info.soc_name = soc.soc_name
        info.soc_vendor = soc.vendor
        info.cpu_arch = soc.cpu_arch
        info.cpu_cores = soc.cpu_cores
        info.cpu_max_mhz = soc.cpu_max_mhz
        info.gpu_name = soc.gpu_name
        info.gpu_cuda_cores = soc.gpu_cuda_cores
        info.npu_name = soc.npu_name
        info.npu_tops = soc.npu_tops
        info.process_nm = soc.process_nm

    # ── Determine family ──
    info.family = _detect_family(model, compatible)

    # ── RAM ──
    info.ram_mb = _detect_ram_mb()

    # ── GPIO ──
    info.has_gpio, info.gpio_pins = _detect_gpio()

    # ── Peripherals ──
    _detect_peripherals(info)

    # ── Jetson-specific ──
    if info.family == "jetson" or is_jetson():
        info.family = "jetson"
        _detect_jetson_extras(info)

    # ── Raspberry Pi specific ──
    if info.family == "raspberry_pi":
        _detect_rpi_extras(info, model)

    # ── Fan detection ──
    info.has_fan = _detect_fan()

    _sbc_info = info
    log.info("SBC detected: %s (%s)", info.board_name, info.soc_name)
    return info


def _lookup_sbc_soc(model: str, compatible: str) -> Optional[_SBCSoC]:
    """Match device-tree model / compatible against SBC SoC database."""
    search_text = f"{model} {compatible}".lower()

    # Exact key matches first
    for key, soc in _SBC_SOC_DATABASE.items():
        if key in search_text:
            return soc

    return None


def _detect_family(model: str, compatible: str) -> str:
    """Determine the SBC family from model / compatible strings."""
    search = f"{model} {compatible}".lower()
    for pattern, family in _BOARD_FAMILIES.items():
        if pattern in search:
            return family
    return "generic"


def _detect_ram_mb() -> int:
    """Detect total RAM in megabytes."""
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        pass
    # Fallback: /proc/meminfo
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                kb = int(line.split()[1])
                return kb // 1024
    except Exception:
        pass
    return 0


def _detect_gpio() -> Tuple[bool, int]:
    """Detect GPIO availability and pin count."""
    has_gpio = False
    pins = 0

    # Check /sys/class/gpio
    try:
        gpio_path = Path("/sys/class/gpio")
        if gpio_path.exists():
            has_gpio = True
            # Count exported GPIO lines
            exports = list(gpio_path.glob("gpio[0-9]*"))
            if exports:
                pins = len(exports)
    except Exception:
        pass

    # Check for RPi.GPIO / gpiod availability
    if not has_gpio:
        try:
            import RPi.GPIO  # type: ignore[import-untyped]
            has_gpio = True
            pins = 40
        except ImportError:
            pass
        try:
            gpiochips = list(Path("/dev").glob("gpiochip*"))
            if gpiochips:
                has_gpio = True
        except Exception:
            pass

    # Known pin counts by family
    if has_gpio and pins == 0:
        model = _read_dt_model().lower()
        if "raspberry pi" in model:
            pins = 40
        elif "beaglebone" in model:
            pins = 65  # 2×46 header minus power/ground
        elif "jetson" in model:
            pins = 40
        elif "orange pi" in model:
            pins = 40

    return has_gpio, pins


def _detect_peripherals(info: SBCInfo) -> None:
    """Detect common SBC peripherals."""
    # WiFi
    try:
        wifi_path = Path("/sys/class/net")
        if wifi_path.exists():
            for iface in wifi_path.iterdir():
                if (iface / "wireless").exists() or iface.name.startswith("wlan"):
                    info.has_wifi = True
                    break
    except Exception:
        pass

    # Bluetooth
    try:
        bt_path = Path("/sys/class/bluetooth")
        info.has_bluetooth = bt_path.exists() and any(bt_path.iterdir())
    except Exception:
        pass

    # Ethernet
    try:
        net_path = Path("/sys/class/net")
        if net_path.exists():
            for iface in net_path.iterdir():
                name = iface.name
                if name.startswith("eth") or name.startswith("enp") or name.startswith("end"):
                    info.has_ethernet = True
                    break
    except Exception:
        pass

    # USB ports (count USB host controllers)
    try:
        usb_path = Path("/sys/bus/usb/devices")
        if usb_path.exists():
            # Count root hubs (usb1, usb2, ...)
            info.usb_ports = len(list(usb_path.glob("usb[0-9]*")))
    except Exception:
        pass

    # PCIe
    try:
        pci_path = Path("/sys/bus/pci/devices")
        info.has_pcie = pci_path.exists() and any(pci_path.iterdir())
    except Exception:
        pass

    # Camera CSI
    try:
        csi_paths = [
            Path("/dev/video0"),
            Path("/sys/class/video4linux"),
        ]
        info.has_camera_csi = any(p.exists() for p in csi_paths)
    except Exception:
        pass

    # Display DSI
    try:
        dsi_paths = [
            Path("/sys/class/drm"),
            Path("/dev/fb0"),
        ]
        info.has_display_dsi = any(p.exists() for p in dsi_paths)
    except Exception:
        pass

    # Storage type detection
    info.storage_type = _detect_storage_type()


def _detect_storage_type() -> str:
    """Detect primary storage type."""
    try:
        # Check for NVMe
        if Path("/sys/class/nvme").exists() and list(Path("/sys/class/nvme").iterdir()):
            return "NVMe"
        # Check for eMMC
        if Path("/sys/class/mmc_host").exists():
            for host in Path("/sys/class/mmc_host").iterdir():
                for dev in host.glob("mmc*"):
                    try:
                        dev_type = (dev / "type").read_text().strip()
                        if dev_type == "MMC":
                            return "eMMC"
                        elif dev_type == "SD":
                            return "microSD"
                    except Exception:
                        pass
        # Check for SD via block devices
        block_path = Path("/sys/class/block")
        if block_path.exists():
            for dev in block_path.iterdir():
                name = dev.name
                if name.startswith("mmcblk"):
                    return "microSD/eMMC"
    except Exception:
        pass
    return ""


def _detect_fan() -> bool:
    """Detect if a cooling fan is present."""
    try:
        # Check hwmon for fan sensors
        hwmon = Path("/sys/class/hwmon")
        if hwmon.exists():
            for hw in hwmon.iterdir():
                for f in hw.glob("fan*_input"):
                    return True
        # Check for thermal cooling fans
        thermal = Path("/sys/class/thermal")
        if thermal.exists():
            for cool in thermal.glob("cooling_device*"):
                try:
                    cool_type = (cool / "type").read_text().strip().lower()
                    if "fan" in cool_type:
                        return True
                except Exception:
                    pass
        # Jetson fan control
        if Path("/sys/devices/pwm-fan").exists():
            return True
    except Exception:
        pass
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Jetson Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _detect_jetson_extras(info: SBCInfo) -> None:
    """Detect NVIDIA Jetson-specific information."""
    # L4T version from /etc/nv_tegra_release
    try:
        tegra = Path("/etc/nv_tegra_release").read_text()
        m = re.search(r"R(\d+)\s+\(release\),\s+REVISION:\s+(\d+\.\d+)", tegra)
        if m:
            info.jetson_l4t_version = f"{m.group(1)}.{m.group(2)}"
    except Exception:
        pass

    # Power mode from nvpmodel
    try:
        r = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "NV Power Mode" in line or "POWER_MODEL" in line:
                    info.jetson_power_mode = line.split(":")[-1].strip()
                    break
            if not info.jetson_power_mode:
                # Try to parse the mode name from output
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if line and not line.startswith("NV") and not line.startswith("WARN"):
                        info.jetson_power_mode = line
                        break
    except Exception:
        pass


def _detect_rpi_extras(info: SBCInfo, model: str) -> None:
    """Detect Raspberry Pi-specific information."""
    # Raspberry Pi always has 40-pin GPIO (Except Pi Pico, Zero W has 40 too)
    if info.gpio_pins == 0:
        info.has_gpio = True
        info.gpio_pins = 40

    # Camera CSI is standard on most Pi models
    info.has_camera_csi = True
    info.has_display_dsi = True

    # Storage is always microSD (primary), may have USB boot
    if not info.storage_type:
        info.storage_type = "microSD"

    # Pi 4 and Pi 5 have USB 3.0 ports
    ml = model.lower()
    if "pi 5" in ml:
        info.has_pcie = True  # Pi 5 has PCIe x1
    elif "pi 4" in ml:
        info.has_pcie = False  # Pi 4 has VL805 USB 3 (internal PCIe, not exposed)


# ═══════════════════════════════════════════════════════════════════════════
#  Worker Recommendations for IoT
# ═══════════════════════════════════════════════════════════════════════════

def recommend_iot_workers(io_bound: bool = False) -> int:
    """Recommend worker count for memory-constrained SBC environments.

    Rules:
    - MicroPython/CircuitPython: always 1 (single-threaded runtimes)
    - RAM < 256 MB: max 1 worker
    - RAM < 512 MB: max 2 workers
    - RAM < 1024 MB: max cpu_cores (capped at 4)
    - RAM < 2048 MB: cpu_cores for CPU-bound, cpu_cores×2 for I/O-bound
    - RAM >= 2048 MB: standard sizing (cpu_cores or cpu_cores×2)
    """
    if is_micropython() or is_circuitpython():
        return 1

    ram = _detect_ram_mb()
    try:
        cores = os.cpu_count() or 1
    except Exception:
        cores = 1

    if ram > 0 and ram < 256:
        return 1
    if ram > 0 and ram < 512:
        return min(2, cores)
    if ram > 0 and ram < 1024:
        return min(4, cores)
    if ram > 0 and ram < 2048:
        if io_bound:
            return min(cores * 2, 8)
        return min(cores, 4)

    # >= 2GB: standard
    if io_bound:
        return min(cores * 2, 16)
    return cores


# ═══════════════════════════════════════════════════════════════════════════
#  Thermal Monitoring for SBCs
# ═══════════════════════════════════════════════════════════════════════════

def get_sbc_thermal() -> Dict[str, Any]:
    """Read thermal info relevant to SBC operation.

    Returns dict with cpu_temp, gpu_temp (if available), is_throttled,
    fan_rpm (if fan present), and thermal recommendation.
    """
    result: Dict[str, Any] = {
        "cpu_temp_c": None,
        "gpu_temp_c": None,
        "is_throttled": False,
        "fan_rpm": None,
        "recommendation": "normal",
    }

    try:
        from pyaccelerate.android import get_thermal_zones
        temps = get_thermal_zones()

        for name, temp in temps.items():
            nl = name.lower()
            if result["cpu_temp_c"] is None and any(k in nl for k in ("cpu", "soc", "thermal_zone0")):
                result["cpu_temp_c"] = temp
            if result["gpu_temp_c"] is None and any(k in nl for k in ("gpu", "video")):
                result["gpu_temp_c"] = temp
    except Exception:
        pass

    # Throttle check
    cpu_temp = result["cpu_temp_c"]
    if cpu_temp is not None:
        if cpu_temp > 80:
            result["is_throttled"] = True
            result["recommendation"] = "critical — reduce workload"
        elif cpu_temp > 70:
            result["recommendation"] = "warm — consider reducing workers"
        elif cpu_temp > 60:
            result["recommendation"] = "moderate"

    # Jetson-specific: check for DVFS throttling
    try:
        throttle_path = Path("/sys/devices/gpu.0/railgate_enable")
        if throttle_path.exists():
            val = throttle_path.read_text().strip()
            if val == "1":
                result["is_throttled"] = True
    except Exception:
        pass

    # Fan RPM
    try:
        hwmon = Path("/sys/class/hwmon")
        if hwmon.exists():
            for hw in hwmon.iterdir():
                for f in hw.glob("fan*_input"):
                    result["fan_rpm"] = int(f.read_text().strip())
                    break
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Jetson Power Model Control
# ═══════════════════════════════════════════════════════════════════════════

def get_jetson_power_modes() -> List[Dict[str, str]]:
    """List available Jetson power modes.

    Returns list of dicts with 'id', 'name', 'active' keys.
    """
    modes: List[Dict[str, str]] = []
    if not is_jetson():
        return modes

    try:
        r = subprocess.run(
            ["nvpmodel", "-p", "--verbose"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            current = ""
            # Get current mode
            r2 = subprocess.run(
                ["nvpmodel", "-q"],
                capture_output=True, text=True, timeout=5,
            )
            if r2.returncode == 0:
                for line in r2.stdout.splitlines():
                    if "NV Power Mode" in line:
                        current = line.split(":")[-1].strip()

            for line in r.stdout.splitlines():
                m = re.match(r"\s*(\d+)\s+(.+)", line)
                if m:
                    mode_id = m.group(1)
                    mode_name = m.group(2).strip()
                    modes.append({
                        "id": mode_id,
                        "name": mode_name,
                        "active": "yes" if mode_name == current else "no",
                    })
    except Exception:
        pass
    return modes


def set_jetson_power_mode(mode_id: int) -> bool:
    """Set Jetson power mode.  Requires root/sudo."""
    if not is_jetson():
        return False
    try:
        r = subprocess.run(
            ["sudo", "nvpmodel", "-m", str(mode_id)],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  Coral Edge TPU Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_coral_tpu() -> Optional[Dict[str, str]]:
    """Detect Google Coral Edge TPU (USB or PCIe).

    Returns dict with 'type' (usb/pcie), 'name', 'available' keys,
    or None if not detected.
    """
    # USB Coral (Vendor ID 1a6e / 18d1, Product ID 089a)
    try:
        r = subprocess.run(
            ["lsusb"], capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if "18d1:089a" in line or "1a6e:089a" in line or "Google" in line and "Coral" in line:
                    has_runtime = _check_edgetpu_runtime()
                    return {
                        "type": "usb",
                        "name": "Coral USB Accelerator",
                        "tops": "4.0",
                        "runtime": "yes" if has_runtime else "no",
                    }
    except Exception:
        pass

    # PCIe Coral (M.2 or mini-PCIe)
    try:
        pci_path = Path("/sys/bus/pci/devices")
        if pci_path.exists():
            for dev in pci_path.iterdir():
                try:
                    vendor = (dev / "vendor").read_text().strip()
                    device = (dev / "device").read_text().strip()
                    # Google Coral PCIe: vendor 1ac1, device 089a
                    if vendor == "0x1ac1" or (vendor == "0x18d1" and device == "0x089a"):
                        has_runtime = _check_edgetpu_runtime()
                        return {
                            "type": "pcie",
                            "name": "Coral Edge TPU (PCIe)",
                            "tops": "4.0",
                            "runtime": "yes" if has_runtime else "no",
                        }
                except Exception:
                    pass
    except Exception:
        pass

    return None


def _check_edgetpu_runtime() -> bool:
    """Check if Edge TPU runtime / pycoral is installed."""
    try:
        import pycoral  # type: ignore[import-untyped]
        return True
    except ImportError:
        pass
    try:
        import tflite_runtime  # type: ignore[import-untyped]
        return True
    except ImportError:
        pass
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  Public Convenience API
# ═══════════════════════════════════════════════════════════════════════════

def get_sbc_summary() -> Dict[str, Any]:
    """Machine-readable summary of SBC information."""
    info = detect_sbc()
    if not info:
        return {"is_sbc": False}

    return {
        "is_sbc": True,
        "board_name": info.board_name,
        "family": info.family,
        "soc_name": info.soc_name,
        "soc_vendor": info.soc_vendor,
        "cpu_arch": info.cpu_arch,
        "cpu_cores": info.cpu_cores,
        "cpu_max_mhz": info.cpu_max_mhz,
        "ram_mb": info.ram_mb,
        "gpu_name": info.gpu_name,
        "gpu_cuda_cores": info.gpu_cuda_cores,
        "npu_name": info.npu_name,
        "npu_tops": info.npu_tops,
        "has_gpio": info.has_gpio,
        "gpio_pins": info.gpio_pins,
        "has_pcie": info.has_pcie,
        "has_wifi": info.has_wifi,
        "has_bluetooth": info.has_bluetooth,
        "has_ethernet": info.has_ethernet,
        "usb_ports": info.usb_ports,
        "storage_type": info.storage_type,
        "has_fan": info.has_fan,
        "jetson_power_mode": info.jetson_power_mode,
        "jetson_l4t_version": info.jetson_l4t_version,
        "recommended_workers": recommend_iot_workers(),
    }


def reset_cache() -> None:
    """Reset all cached SBC detection data."""
    global _is_sbc, _is_micropython, _is_circuitpython, _sbc_info
    global _dt_model_cache, _dt_compat_cache
    _is_sbc = None
    _is_micropython = None
    _is_circuitpython = None
    _sbc_info = None
    _dt_model_cache = None
    _dt_compat_cache = None
