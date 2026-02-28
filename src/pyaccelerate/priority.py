"""
pyaccelerate.priority — OS-level task priority & energy profile management.

Controls how the operating system schedules the current process and its
threads.  Provides two orthogonal axes:

1. **Task priority** — CPU scheduling priority (``idle → realtime``).
2. **Energy profile** — power/performance trade-off on supported OSes.

Supported platforms:
  - **Windows**: ``SetPriorityClass`` + ``powercfg``
  - **Linux**: ``nice``, ``ionice``, ``cpufreq`` governor
  - **macOS**: ``nice``, ``pmset`` assertions

Thread-safe. All changes affect the *current process* only.

Usage::

    from pyaccelerate.priority import (
        TaskPriority, EnergyProfile,
        set_task_priority, set_energy_profile,
        get_task_priority, get_energy_profile,
        max_performance, balanced, power_saver,
    )

    # Maximize performance
    max_performance()

    # Or fine-grained control
    set_task_priority(TaskPriority.HIGH)
    set_energy_profile(EnergyProfile.PERFORMANCE)
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from enum import IntEnum, auto
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("pyaccelerate.priority")


# ═══════════════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════════════

class TaskPriority(IntEnum):
    """Process scheduling priority levels (OS-agnostic)."""
    IDLE = 0           # Lowest — only runs when system is idle
    BELOW_NORMAL = 1   # Below default
    NORMAL = 2         # Default OS priority
    ABOVE_NORMAL = 3   # Above default
    HIGH = 4           # High priority
    REALTIME = 5       # Highest — use with extreme caution


class EnergyProfile(IntEnum):
    """System energy / performance profile."""
    POWER_SAVER = 0      # Minimize power, clocks may be reduced
    BALANCED = 1          # OS default trade-off
    PERFORMANCE = 2       # Maximum performance, maximum power
    ULTRA_PERFORMANCE = 3 # Disable throttling, turbo always on


# ═══════════════════════════════════════════════════════════════════════════
#  Task Priority
# ═══════════════════════════════════════════════════════════════════════════

def set_task_priority(priority: TaskPriority) -> bool:
    """Set the scheduling priority of the current process.

    Returns True on success, False on failure (e.g. insufficient permissions).
    """
    system = platform.system()
    try:
        if system == "Windows":
            return _set_priority_windows(priority)
        elif system == "Linux":
            return _set_priority_linux(priority)
        elif system == "Darwin":
            return _set_priority_darwin(priority)
        else:
            log.warning("Unsupported OS for priority: %s", system)
            return False
    except Exception as exc:
        log.warning("Failed to set task priority %s: %s", priority.name, exc)
        return False


def get_task_priority() -> TaskPriority:
    """Get the current process scheduling priority."""
    system = platform.system()
    try:
        if system == "Windows":
            return _get_priority_windows()
        elif system in ("Linux", "Darwin"):
            return _get_priority_unix()
    except Exception:
        pass
    return TaskPriority.NORMAL


def set_io_priority(high: bool = True) -> bool:
    """Set I/O scheduling priority (Linux only).

    Parameters
    ----------
    high : bool
        True = best-effort class 0 (highest), False = best-effort class 7 (lowest).
    """
    system = platform.system()
    if system != "Linux":
        return False
    try:
        import psutil  # type: ignore[import-untyped]
        p = psutil.Process()
        if high:
            p.ionice(psutil.IOPRIO_CLASS_BE, value=0)
        else:
            p.ionice(psutil.IOPRIO_CLASS_BE, value=7)
        log.info("I/O priority set to %s", "high" if high else "low")
        return True
    except Exception as exc:
        log.debug("Failed to set I/O priority: %s", exc)
        return False


# ── Windows ─────────────────────────────────────────────────────────────

def _set_priority_windows(priority: TaskPriority) -> bool:
    """Set process priority on Windows via psutil or ctypes."""
    try:
        import psutil  # type: ignore[import-untyped]

        # Map our levels to psutil/Windows priority classes
        _MAP = {
            TaskPriority.IDLE: psutil.IDLE_PRIORITY_CLASS,
            TaskPriority.BELOW_NORMAL: psutil.BELOW_NORMAL_PRIORITY_CLASS,
            TaskPriority.NORMAL: psutil.NORMAL_PRIORITY_CLASS,
            TaskPriority.ABOVE_NORMAL: psutil.ABOVE_NORMAL_PRIORITY_CLASS,
            TaskPriority.HIGH: psutil.HIGH_PRIORITY_CLASS,
            TaskPriority.REALTIME: psutil.REALTIME_PRIORITY_CLASS,
        }
        p = psutil.Process()
        p.nice(_MAP[priority])
        log.info("Windows process priority set to %s", priority.name)
        return True
    except Exception as exc:
        log.warning("Windows priority failed: %s", exc)
        return False


def _get_priority_windows() -> TaskPriority:
    try:
        import psutil  # type: ignore[import-untyped]
        p = psutil.Process()
        nice = p.nice()
        _REV = {
            psutil.IDLE_PRIORITY_CLASS: TaskPriority.IDLE,
            psutil.BELOW_NORMAL_PRIORITY_CLASS: TaskPriority.BELOW_NORMAL,
            psutil.NORMAL_PRIORITY_CLASS: TaskPriority.NORMAL,
            psutil.ABOVE_NORMAL_PRIORITY_CLASS: TaskPriority.ABOVE_NORMAL,
            psutil.HIGH_PRIORITY_CLASS: TaskPriority.HIGH,
            psutil.REALTIME_PRIORITY_CLASS: TaskPriority.REALTIME,
        }
        return _REV.get(nice, TaskPriority.NORMAL)
    except Exception:
        return TaskPriority.NORMAL


# ── Linux / macOS ──────────────────────────────────────────────────────

def _set_priority_linux(priority: TaskPriority) -> bool:
    """Set process priority on Linux via nice value."""
    _NICE_MAP = {
        TaskPriority.IDLE: 19,
        TaskPriority.BELOW_NORMAL: 10,
        TaskPriority.NORMAL: 0,
        TaskPriority.ABOVE_NORMAL: -5,
        TaskPriority.HIGH: -10,
        TaskPriority.REALTIME: -20,
    }
    nice_val = _NICE_MAP[priority]
    try:
        os.nice(nice_val - os.nice(0))
        log.info("Linux nice set to %d (%s)", nice_val, priority.name)
        return True
    except OSError:
        try:
            import psutil  # type: ignore[import-untyped]
            p = psutil.Process()
            p.nice(nice_val)
            return True
        except Exception as exc:
            log.warning("Linux priority failed: %s", exc)
            return False


def _set_priority_darwin(priority: TaskPriority) -> bool:
    """Set process priority on macOS via nice."""
    return _set_priority_linux(priority)  # same mechanism


def _get_priority_unix() -> TaskPriority:
    try:
        nice = os.nice(0)
        if nice >= 15:
            return TaskPriority.IDLE
        elif nice >= 5:
            return TaskPriority.BELOW_NORMAL
        elif nice >= -2:
            return TaskPriority.NORMAL
        elif nice >= -7:
            return TaskPriority.ABOVE_NORMAL
        elif nice >= -15:
            return TaskPriority.HIGH
        else:
            return TaskPriority.REALTIME
    except Exception:
        return TaskPriority.NORMAL


# ═══════════════════════════════════════════════════════════════════════════
#  Energy Profile
# ═══════════════════════════════════════════════════════════════════════════

# Well-known Windows power plan GUIDs
_WIN_POWER_PLANS = {
    EnergyProfile.POWER_SAVER: "a1841308-3541-4fab-bc81-f71556f20b4a",
    EnergyProfile.BALANCED: "381b4222-f694-41f0-9685-ff5bb260df2e",
    EnergyProfile.PERFORMANCE: "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
    EnergyProfile.ULTRA_PERFORMANCE: "e9a42b02-d5df-448d-aa00-03f14749eb61",
}


def set_energy_profile(profile: EnergyProfile) -> bool:
    """Set the system energy/performance profile.

    On Windows, changes the active power plan.
    On Linux, sets the CPU frequency governor.
    On macOS, creates/releases a power assertion.

    Returns True on success.
    """
    system = platform.system()
    try:
        if system == "Windows":
            return _set_energy_windows(profile)
        elif system == "Linux":
            return _set_energy_linux(profile)
        elif system == "Darwin":
            return _set_energy_darwin(profile)
        else:
            log.warning("Unsupported OS for energy profile: %s", system)
            return False
    except Exception as exc:
        log.warning("Failed to set energy profile %s: %s", profile.name, exc)
        return False


def get_energy_profile() -> EnergyProfile:
    """Get the current energy profile (best effort)."""
    system = platform.system()
    try:
        if system == "Windows":
            return _get_energy_windows()
        elif system == "Linux":
            return _get_energy_linux()
    except Exception:
        pass
    return EnergyProfile.BALANCED


def _set_energy_windows(profile: EnergyProfile) -> bool:
    """Set Windows power plan via powercfg."""
    guid = _WIN_POWER_PLANS.get(profile)
    if not guid:
        return False

    # First ensure the plan exists (Ultra Performance may need enabling)
    if profile == EnergyProfile.ULTRA_PERFORMANCE:
        subprocess.run(
            ["powercfg", "/duplicatescheme", guid],
            capture_output=True, timeout=10,
        )

    r = subprocess.run(
        ["powercfg", "/setactive", guid],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode == 0:
        log.info("Windows power plan set to %s (%s)", profile.name, guid)
        return True
    log.debug("powercfg failed: %s", r.stderr.strip())
    return False


def _get_energy_windows() -> EnergyProfile:
    r = subprocess.run(
        ["powercfg", "/getactivescheme"],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        return EnergyProfile.BALANCED
    out = r.stdout.lower()
    for profile, guid in _WIN_POWER_PLANS.items():
        if guid in out:
            return profile
    return EnergyProfile.BALANCED


def _set_energy_linux(profile: EnergyProfile) -> bool:
    """Set Linux CPU frequency governor."""
    _GOV_MAP = {
        EnergyProfile.POWER_SAVER: "powersave",
        EnergyProfile.BALANCED: "schedutil",
        EnergyProfile.PERFORMANCE: "performance",
        EnergyProfile.ULTRA_PERFORMANCE: "performance",
    }
    governor = _GOV_MAP.get(profile, "schedutil")

    # Try cpupower first
    r = subprocess.run(
        ["cpupower", "frequency-set", "-g", governor],
        capture_output=True, timeout=10,
    )
    if r.returncode == 0:
        log.info("Linux governor set to %s", governor)
        return True

    # Fallback: write directly to sysfs
    try:
        import pathlib
        cpus = list(pathlib.Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"))
        if cpus:
            for gov_file in cpus:
                gov_file.write_text(governor)
            log.info("Linux governor set to %s (sysfs)", governor)
            return True
    except Exception as exc:
        log.debug("sysfs governor failed: %s", exc)

    return False


def _get_energy_linux() -> EnergyProfile:
    try:
        import pathlib
        gov = pathlib.Path(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ).read_text().strip()
        _REV = {
            "powersave": EnergyProfile.POWER_SAVER,
            "conservative": EnergyProfile.POWER_SAVER,
            "schedutil": EnergyProfile.BALANCED,
            "ondemand": EnergyProfile.BALANCED,
            "performance": EnergyProfile.PERFORMANCE,
        }
        return _REV.get(gov, EnergyProfile.BALANCED)
    except Exception:
        return EnergyProfile.BALANCED


def _set_energy_darwin(profile: EnergyProfile) -> bool:
    """macOS energy via caffeinate / pmset assertion."""
    if profile in (EnergyProfile.PERFORMANCE, EnergyProfile.ULTRA_PERFORMANCE):
        # Prevent sleep / throttling
        r = subprocess.run(
            ["pmset", "-a", "lowpowermode", "0"],
            capture_output=True, timeout=10,
        )
        log.info("macOS lowpowermode disabled")
        return r.returncode == 0
    elif profile == EnergyProfile.POWER_SAVER:
        r = subprocess.run(
            ["pmset", "-a", "lowpowermode", "1"],
            capture_output=True, timeout=10,
        )
        log.info("macOS lowpowermode enabled")
        return r.returncode == 0
    return True  # balanced = no change


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience presets
# ═══════════════════════════════════════════════════════════════════════════

def max_performance() -> Dict[str, bool]:
    """Set maximum performance on all axes.

    - Task priority → HIGH
    - Energy profile → ULTRA_PERFORMANCE
    - I/O priority → high (Linux)

    Returns dict of results.
    """
    return {
        "task_priority": set_task_priority(TaskPriority.HIGH),
        "energy_profile": set_energy_profile(EnergyProfile.ULTRA_PERFORMANCE),
        "io_priority": set_io_priority(high=True),
    }


def balanced() -> Dict[str, bool]:
    """Reset to balanced defaults.

    - Task priority → NORMAL
    - Energy profile → BALANCED
    """
    return {
        "task_priority": set_task_priority(TaskPriority.NORMAL),
        "energy_profile": set_energy_profile(EnergyProfile.BALANCED),
    }


def power_saver() -> Dict[str, bool]:
    """Minimize power consumption.

    - Task priority → BELOW_NORMAL
    - Energy profile → POWER_SAVER
    - I/O priority → low (Linux)
    """
    return {
        "task_priority": set_task_priority(TaskPriority.BELOW_NORMAL),
        "energy_profile": set_energy_profile(EnergyProfile.POWER_SAVER),
        "io_priority": set_io_priority(high=False),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Info
# ═══════════════════════════════════════════════════════════════════════════

def get_priority_info() -> Dict[str, str]:
    """Get a summary of current priority/energy settings."""
    return {
        "task_priority": get_task_priority().name,
        "energy_profile": get_energy_profile().name,
        "os": platform.system(),
    }
