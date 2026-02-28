"""
pyaccelerate.gpu.intel — Intel oneAPI / dpctl compute helpers.

Provides GPU-accelerated operations via Intel's Data Parallel Control (dpctl)
and Data Parallel NumPy (dpnp) libraries. Supports Intel UHD, Iris, and Arc
(discrete) GPUs.

All functions gracefully degrade when dpctl/dpnp is not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("pyaccelerate.gpu.intel")

_dpctl: Any = None
_dpnp: Any = None
_available: Optional[bool] = None


def is_available() -> bool:
    """True if dpctl is installed and at least one Intel GPU is found."""
    global _dpctl, _dpnp, _available
    if _available is not None:
        return _available
    try:
        import dpctl  # type: ignore[import-untyped]
        for dev in dpctl.get_devices():
            if dev.device_type.name == "gpu":
                _dpctl = dpctl
                _available = True
                # Try dpnp as well (optional)
                try:
                    import dpnp  # type: ignore[import-untyped]
                    _dpnp = dpnp
                except ImportError:
                    pass
                return True
        _available = False
    except Exception:
        _available = False
    return _available


def get_dpctl() -> Any:
    """Return the ``dpctl`` module (or raise ImportError)."""
    if not is_available():
        raise ImportError("dpctl not available or no Intel GPU found")
    return _dpctl


def get_dpnp() -> Any:
    """Return the ``dpnp`` module (or raise ImportError)."""
    if _dpnp is None:
        raise ImportError("dpnp not installed")
    return _dpnp


def list_devices() -> List[Dict[str, Any]]:
    """List all Intel GPU devices via dpctl."""
    dpctl = get_dpctl()
    devices: List[Dict[str, Any]] = []
    for dev in dpctl.get_devices():
        if dev.device_type.name != "gpu":
            continue
        info: Dict[str, Any] = {"name": dev.name}
        try:
            info["memory_bytes"] = dev.global_mem_size
        except Exception:
            info["memory_bytes"] = 0
        try:
            info["max_compute_units"] = dev.max_compute_units
        except Exception:
            pass
        devices.append(info)
    return devices


def to_device(data: Any) -> Any:
    """Transfer a numpy array to Intel GPU via dpnp."""
    dpnp = get_dpnp()
    import numpy as np  # type: ignore[import-untyped]
    if isinstance(data, (bytes, bytearray)):
        data = np.frombuffer(bytearray(data), dtype=np.uint8)
    return dpnp.asarray(data)


def to_host(gpu_array: Any) -> Any:
    """Transfer a dpnp array back to host (numpy)."""
    import numpy as np  # type: ignore[import-untyped]
    return np.asarray(gpu_array)
