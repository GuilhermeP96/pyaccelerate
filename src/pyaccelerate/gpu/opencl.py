"""
pyaccelerate.gpu.opencl — PyOpenCL compute helpers.

Provides GPU-accelerated operations via PyOpenCL when available:
  - Context / queue management per device
  - Buffer creation & host ↔ device transfer
  - Kernel compilation and execution helpers

All functions gracefully degrade when PyOpenCL is not installed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("pyaccelerate.gpu.opencl")

_cl: Any = None
_available: Optional[bool] = None


def is_available() -> bool:
    """True if PyOpenCL is installed and at least one GPU platform exists."""
    global _cl, _available
    if _available is not None:
        return _available
    try:
        import pyopencl as cl  # type: ignore[import-untyped]
        platforms = cl.get_platforms()
        for p in platforms:
            try:
                devs = p.get_devices(device_type=cl.device_type.GPU)
                if devs:
                    _cl = cl
                    _available = True
                    return True
            except Exception:
                continue
        _available = False
    except Exception:
        _available = False
    return _available


def get_module() -> Any:
    """Return the ``pyopencl`` module (or raise ImportError)."""
    if not is_available():
        raise ImportError("PyOpenCL not available or no GPU platforms found")
    return _cl


def list_devices() -> List[Dict[str, Any]]:
    """List all OpenCL GPU devices across all platforms."""
    cl = get_module()
    devices: List[Dict[str, Any]] = []
    for plat in cl.get_platforms():
        try:
            for dev in plat.get_devices(device_type=cl.device_type.GPU):
                devices.append({
                    "name": dev.name.strip(),
                    "platform": plat.name,
                    "vendor": dev.vendor,
                    "memory_bytes": dev.global_mem_size,
                    "compute_units": dev.max_compute_units,
                    "max_work_group_size": dev.max_work_group_size,
                    "version": dev.version,
                })
        except Exception:
            continue
    return devices


class Context:
    """Managed OpenCL context + command queue for a specific device.

    Usage::

        with Context(device_name="NVIDIA GeForce RTX 4090") as ctx:
            buf = ctx.create_buffer(data)
            ctx.enqueue_read(buf, out_array)
    """

    def __init__(self, device_name: Optional[str] = None, device_index: int = 0):
        cl = get_module()
        self._cl = cl
        self._ctx: Any = None
        self._queue: Any = None
        self._device: Any = None

        # Find the requested device
        for plat in cl.get_platforms():
            try:
                devs = plat.get_devices(device_type=cl.device_type.GPU)
            except Exception:
                continue
            for dev in devs:
                if device_name and device_name.lower() not in dev.name.lower():
                    continue
                self._device = dev
                break
            if self._device:
                break

        if self._device is None:
            raise RuntimeError(
                f"OpenCL GPU device not found: {device_name or f'index {device_index}'}"
            )

    def __enter__(self) -> "Context":
        cl = self._cl
        self._ctx = cl.Context(devices=[self._device])
        self._queue = cl.CommandQueue(self._ctx)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._queue is not None:
            self._queue.finish()

    def create_buffer(
        self,
        host_data: Any,
        read_only: bool = True,
    ) -> Any:
        """Create a device buffer from host data (numpy array or bytes)."""
        cl = self._cl
        import numpy as np  # type: ignore[import-untyped]

        if isinstance(host_data, (bytes, bytearray)):
            host_data = np.frombuffer(bytearray(host_data), dtype=np.uint8)

        flags = cl.mem_flags.READ_ONLY if read_only else cl.mem_flags.READ_WRITE
        flags |= cl.mem_flags.COPY_HOST_PTR
        return cl.Buffer(self._ctx, flags, hostbuf=host_data)

    def read_buffer(self, buf: Any, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Read a device buffer back to host as a numpy array."""
        import numpy as np  # type: ignore[import-untyped]

        if dtype is None:
            dtype = np.float32
        out = np.empty(shape, dtype=dtype)
        self._cl.enqueue_copy(self._queue, out, buf)
        self._queue.finish()
        return out

    def compile_kernel(self, source: str) -> Any:
        """Compile an OpenCL kernel program from source string."""
        return self._cl.Program(self._ctx, source).build()

    @property
    def device(self) -> Any:
        return self._device

    @property
    def queue(self) -> Any:
        return self._queue

    @property
    def ctx(self) -> Any:
        return self._ctx
