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
- **priority**   : OS-level task priority & energy profile management
- **max_mode**   : Maximum optimization mode — all resources in parallel
- **android**    : Android/Termux platform detection, ARM SoC database, thermal/battery
- **engine**     : Unified orchestrator that auto-tunes all subsystems

Quick start::

    from pyaccelerate import Engine

    engine = Engine()          # auto-detects hardware
    print(engine.summary())    # human-readable report

    # Maximum optimization mode
    from pyaccelerate.max_mode import MaxMode

    with MaxMode() as m:
        results = m.run_all(
            cpu_fn=cpu_task, cpu_items=cpu_data,
            io_fn=io_task, io_items=io_data,
        )

    # OS priority control
    from pyaccelerate.priority import max_performance, balanced
    max_performance()   # HIGH priority + ULTRA energy
    balanced()          # restore defaults

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
from pyaccelerate.max_mode import MaxMode  # noqa: E402, F401
from pyaccelerate.priority import (  # noqa: E402, F401
    TaskPriority,
    EnergyProfile,
    max_performance,
    balanced,
    power_saver,
)

__all__ = [
    "__version__",
    "Engine",
    "MaxMode",
    "TaskPriority",
    "EnergyProfile",
    "max_performance",
    "balanced",
    "power_saver",
]
