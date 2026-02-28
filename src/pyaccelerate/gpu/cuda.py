"""
pyaccelerate.gpu.cuda — CUDA / CuPy compute helpers.

Provides GPU-accelerated operations via CuPy when available:
  - Array creation & transfer (host ↔ device)
  - Element-wise and reduction kernels
  - Stream management for async overlap
  - Memory pool statistics

All functions gracefully fall back to NumPy / CPU when CuPy is not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

log = logging.getLogger("pyaccelerate.gpu.cuda")

_cp: Any = None
_available: Optional[bool] = None


def is_available() -> bool:
    """True if CuPy + CUDA runtime is usable."""
    global _cp, _available
    if _available is not None:
        return _available
    try:
        import cupy as cp  # type: ignore[import-untyped]
        _ = cp.cuda.runtime.getDeviceCount()
        _cp = cp
        _available = True
    except Exception:
        _available = False
    return _available


def get_module() -> Any:
    """Return the ``cupy`` module (or raise ImportError)."""
    if not is_available():
        raise ImportError("CuPy / CUDA not available on this system")
    return _cp


def to_device(data: Any, device_id: int = 0) -> Any:
    """Transfer a numpy array (or bytes) to GPU memory.

    Returns a CuPy array on the specified CUDA device.
    """
    cp = get_module()
    with cp.cuda.Device(device_id):
        if isinstance(data, (bytes, bytearray)):
            return cp.frombuffer(bytearray(data), dtype=cp.uint8)
        return cp.asarray(data)


def to_host(gpu_array: Any) -> Any:
    """Transfer a CuPy array back to host (NumPy array)."""
    cp = get_module()
    return cp.asnumpy(gpu_array)


def device_count() -> int:
    """Number of CUDA devices available."""
    if not is_available():
        return 0
    cp = get_module()
    return cp.cuda.runtime.getDeviceCount()


def device_info(device_id: int = 0) -> dict[str, Any]:
    """Detailed info for a specific CUDA device."""
    cp = get_module()
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    return {
        "name": name,
        "memory_bytes": props.get("totalGlobalMem", 0),
        "sms": props.get("multiProcessorCount", 0),
        "compute_capability": (
            props.get("major", 0),
            props.get("minor", 0),
        ),
    }


def memory_info(device_id: int = 0) -> Tuple[int, int]:
    """Return (free_bytes, total_bytes) for the specified CUDA device."""
    cp = get_module()
    with cp.cuda.Device(device_id):
        free, total = cp.cuda.runtime.memGetInfo()
        return int(free), int(total)


def synchronize(device_id: int = 0) -> None:
    """Block until all operations on the device are complete."""
    cp = get_module()
    with cp.cuda.Device(device_id):
        cp.cuda.Device(device_id).synchronize()


class Stream:
    """Wrapper around a CUDA stream for async overlap.

    Usage::

        with Stream(device_id=0) as s:
            arr = to_device(data)
            # ... kernels on this stream ...
        # stream synchronized on exit
    """

    def __init__(self, device_id: int = 0, non_blocking: bool = True):
        cp = get_module()
        self.device_id = device_id
        self._device_ctx = cp.cuda.Device(device_id)
        self._stream = cp.cuda.Stream(non_blocking=non_blocking)

    def __enter__(self) -> "Stream":
        self._device_ctx.__enter__()
        self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._stream.synchronize()
        self._stream.__exit__(*args)
        self._device_ctx.__exit__(*args)

    @property
    def raw(self) -> Any:
        """Underlying CuPy stream object."""
        return self._stream
