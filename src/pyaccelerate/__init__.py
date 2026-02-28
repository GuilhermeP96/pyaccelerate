"""
PyAccelerate — High-performance Python acceleration engine.

Modules
-------
- **cpu**        : CPU detection, core count, frequency, affinity, NUMA topology
- **threads**    : Virtual thread pool, sliding-window executor, async bridge
- **gpu**        : Multi-vendor GPU detection, ranking, dispatch (CUDA/OpenCL/Intel)
- **npu**        : NPU detection & inference (OpenVINO, ONNX Runtime, DirectML, CoreML)
- **virt**       : Virtualization detection (Hyper-V, VT-x/AMD-V, WSL2, Docker)
- **memory**     : Memory monitoring, pressure detection, pool allocator
- **profiler**   : Decorator-based profiling & timing utilities
- **benchmark**  : Built-in micro-benchmarks for the current host
- **engine**     : Unified orchestrator that auto-tunes all subsystems

Quick start::

    from pyaccelerate import Engine

    engine = Engine()          # auto-detects hardware
    print(engine.summary())    # human-readable report

    # Use the shared virtual-thread pool
    from pyaccelerate.threads import get_pool, run_parallel

    pool = get_pool()
    fut = pool.submit(my_io_func, arg1, arg2)

    # GPU compute
    from pyaccelerate.gpu import detect_all, best_gpu, dispatch

    gpus = detect_all()
    result = dispatch(my_kernel, data, gpus=gpus)
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__: str = _version("pyaccelerate")
except PackageNotFoundError:
    # Running from source / not installed
    from pathlib import Path as _Path

    _vf = _Path(__file__).resolve().parent.parent.parent / "VERSION"
    __version__ = _vf.read_text().strip() if _vf.exists() else "0.0.0-dev"

# Convenience re-exports
from pyaccelerate.engine import Engine  # noqa: E402, F401

__all__ = [
    "__version__",
    "Engine",
]
