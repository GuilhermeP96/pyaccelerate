"""
pyaccelerate.virt — Virtualization & container detection.

Detects hardware virtualization capabilities and container runtimes:
  - Hyper-V (Windows)
  - VT-x / AMD-V / SVM
  - WSL / WSL2
  - Docker / Podman / containerd
  - KVM / QEMU (Linux)
  - Apple Hypervisor Framework (macOS)

Also provides helpers to check if the *current process* is running inside
a VM or container.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("pyaccelerate.virt")


@dataclass
class VirtInfo:
    """Detected virtualization capabilities of the host."""

    # Hardware virtualization
    vtx_enabled: bool = False       # VT-x / AMD-V / SVM
    hyperv_available: bool = False
    hyperv_running: bool = False
    kvm_available: bool = False     # Linux only
    apple_hv: bool = False          # macOS Hypervisor.framework

    # WSL
    wsl_available: bool = False
    wsl_version: int = 0            # 1 or 2

    # Container runtimes
    docker_available: bool = False
    podman_available: bool = False

    # Is *this* process inside a VM / container?
    inside_container: bool = False
    inside_vm: bool = False
    container_runtime: str = ""     # "docker", "podman", "lxc", ""

    platform_name: str = ""

    @property
    def any_hw_virt(self) -> bool:
        """True if any hardware virtualization extension is available."""
        return self.vtx_enabled or self.hyperv_running or self.kvm_available or self.apple_hv

    def summary_parts(self) -> List[str]:
        """Return human-readable capability labels."""
        parts: list[str] = []
        if self.vtx_enabled:
            parts.append("VT-x/AMD-V")
        if self.hyperv_running:
            parts.append("Hyper-V")
        elif self.hyperv_available:
            parts.append("Hyper-V (available)")
        if self.kvm_available:
            parts.append("KVM")
        if self.apple_hv:
            parts.append("Apple HV")
        if self.wsl_available:
            parts.append(f"WSL{self.wsl_version or ''}")
        if self.docker_available:
            parts.append("Docker")
        if self.podman_available:
            parts.append("Podman")
        if self.inside_container:
            parts.append(f"inside container ({self.container_runtime or '?'})")
        if self.inside_vm:
            parts.append("inside VM")
        return parts


# ═══════════════════════════════════════════════════════════════════════════
#  Cached detection
# ═══════════════════════════════════════════════════════════════════════════

_cached: Optional[VirtInfo] = None


def detect() -> VirtInfo:
    """Detect virtualization capabilities. Cached after first call."""
    global _cached
    if _cached is not None:
        return _cached

    vi = VirtInfo(platform_name=platform.system())

    system = platform.system()
    if system == "Windows":
        _detect_windows(vi)
    elif system == "Linux":
        _detect_linux(vi)
    elif system == "Darwin":
        _detect_macos(vi)

    # Container runtimes (cross-platform)
    _detect_container_runtimes(vi)
    _detect_inside_container(vi)

    _cached = vi
    log.info("Virtualization: %s", ", ".join(vi.summary_parts()) or "none detected")
    return vi


def reset_cache() -> None:
    """Force re-detection on next call."""
    global _cached
    _cached = None


# ═══════════════════════════════════════════════════════════════════════════
#  Platform-specific detectors
# ═══════════════════════════════════════════════════════════════════════════

def _detect_windows(vi: VirtInfo) -> None:
    """Detect Hyper-V, VT-x, WSL on Windows."""
    # Hyper-V
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V).State"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            state = r.stdout.strip().lower()
            vi.hyperv_available = state in ("enabled", "enablepending")
            vi.hyperv_running = state == "enabled"
    except Exception:
        pass

    # VT-x / AMD-V
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).VirtualizationFirmwareEnabled"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            vi.vtx_enabled = r.stdout.strip().lower() == "true"
    except Exception:
        pass

    # WSL
    try:
        r = subprocess.run(
            ["wsl", "--status"], capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            vi.wsl_available = True
            vi.wsl_version = 2 if "WSL 2" in r.stdout or "WSL2" in r.stdout else 1
    except Exception:
        pass


def _detect_linux(vi: VirtInfo) -> None:
    """Detect KVM, VT-x/SVM, and VM/container state on Linux."""
    # VT-x / SVM from /proc/cpuinfo
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        vi.vtx_enabled = "vmx" in cpuinfo or "svm" in cpuinfo
    except Exception:
        pass

    # KVM
    vi.kvm_available = Path("/dev/kvm").exists()

    # Are we inside a VM?
    try:
        r = subprocess.run(
            ["systemd-detect-virt", "--vm"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip() != "none":
            vi.inside_vm = True
    except Exception:
        pass


def _detect_macos(vi: VirtInfo) -> None:
    """Detect Apple Hypervisor Framework on macOS."""
    try:
        r = subprocess.run(
            ["sysctl", "-n", "kern.hv_support"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            vi.apple_hv = r.stdout.strip() == "1"
    except Exception:
        pass


def _detect_container_runtimes(vi: VirtInfo) -> None:
    """Check if Docker / Podman CLI tools are available."""
    for tool, attr in [("docker", "docker_available"), ("podman", "podman_available")]:
        try:
            r = subprocess.run(
                [tool, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                setattr(vi, attr, True)
        except Exception:
            pass


def _detect_inside_container(vi: VirtInfo) -> None:
    """Detect if this process is running inside a container."""
    # Check /.dockerenv
    if Path("/.dockerenv").exists():
        vi.inside_container = True
        vi.container_runtime = "docker"
        return

    # Check cgroup
    try:
        cgroup = Path("/proc/1/cgroup").read_text()
        if "docker" in cgroup:
            vi.inside_container = True
            vi.container_runtime = "docker"
            return
        if "lxc" in cgroup:
            vi.inside_container = True
            vi.container_runtime = "lxc"
            return
        if "kubepods" in cgroup:
            vi.inside_container = True
            vi.container_runtime = "kubernetes"
            return
    except Exception:
        pass

    # Check for container env vars
    if os.environ.get("container"):
        vi.inside_container = True
        vi.container_runtime = os.environ.get("container", "unknown")
