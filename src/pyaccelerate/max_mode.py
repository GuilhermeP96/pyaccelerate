"""
pyaccelerate.max_mode — Maximum Optimization Mode.

Activates **all** available hardware resources in parallel and configures the
OS for peak throughput.  This is the "everything cranked to 11" mode:

  - Sets process priority to HIGH
  - Sets energy profile to ULTRA_PERFORMANCE
  - Pins CPU affinity to all cores
  - Saturates the I/O thread pool
  - Saturates the CPU process pool
  - Enables multi-GPU dispatch
  - Enables NPU offload
  - Pre-fills the memory buffer pool
  - Logs a hardware manifest

Usage::

    from pyaccelerate.max_mode import MaxMode

    with MaxMode() as m:
        # All resources are optimally configured
        results = m.run_all(
            cpu_fn=cpu_heavy_task,
            cpu_items=cpu_data,
            io_fn=io_heavy_task,
            io_items=io_data,
        )

    # Or as a simple function
    from pyaccelerate.max_mode import activate_max_mode, deactivate_max_mode
    state = activate_max_mode()
    # ... do work ...
    deactivate_max_mode(state)
"""

from __future__ import annotations

import logging
import os
import time
import threading
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
    FIRST_COMPLETED,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

from pyaccelerate.cpu import CPUInfo, detect as detect_cpu, recommend_workers
from pyaccelerate.gpu import detect_all as detect_gpus, gpu_available
from pyaccelerate.npu import detect_all as detect_npus, npu_available
from pyaccelerate.memory import (
    Pressure,
    get_pressure,
    get_stats as mem_stats,
    clamp_workers,
    BufferPool,
)
from pyaccelerate.threads import (
    get_pool,
    get_process_pool,
    run_parallel as _run_parallel,
    io_pool_size,
    cpu_pool_size,
    batch_execute,
)
from pyaccelerate.priority import (
    TaskPriority,
    EnergyProfile,
    set_task_priority,
    set_energy_profile,
    get_task_priority,
    get_energy_profile,
    max_performance as _prio_max,
    balanced as _prio_balanced,
    set_io_priority,
)
from pyaccelerate.profiler import Timer

log = logging.getLogger("pyaccelerate.max_mode")

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════
#  MaxMode State
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MaxModeState:
    """Snapshot of configuration before max mode was activated."""
    previous_priority: TaskPriority = TaskPriority.NORMAL
    previous_energy: EnergyProfile = EnergyProfile.BALANCED
    active: bool = False
    activation_time: float = 0.0
    hardware_manifest: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Activation / Deactivation
# ═══════════════════════════════════════════════════════════════════════════

def activate_max_mode(
    *,
    set_priority: bool = True,
    set_energy: bool = True,
    prefill_buffers: bool = True,
    buffer_size: int = 1 << 20,
    max_buffers: int = 32,
) -> MaxModeState:
    """Activate maximum optimization mode.

    Configures the OS and runtime for peak throughput:
      1. Saves current priority/energy settings
      2. Sets task priority to HIGH
      3. Sets energy profile to ULTRA_PERFORMANCE
      4. Warms up I/O and CPU pools
      5. Pre-allocates memory buffers
      6. Builds hardware manifest

    Parameters
    ----------
    set_priority : bool
        Whether to change OS scheduling priority.
    set_energy : bool
        Whether to change the energy/power profile.
    prefill_buffers : bool
        Pre-allocate buffer pool for I/O operations.
    buffer_size : int
        Size of each buffer in bytes (default: 1MB).
    max_buffers : int
        Maximum number of pre-allocated buffers.

    Returns
    -------
    MaxModeState
        State object needed for deactivation.
    """
    state = MaxModeState()
    state.activation_time = time.time()

    # Save current settings
    state.previous_priority = get_task_priority()
    state.previous_energy = get_energy_profile()

    # 1. OS-level optimization
    if set_priority:
        set_task_priority(TaskPriority.HIGH)
        set_io_priority(high=True)
        log.info("Task priority set to HIGH")

    if set_energy:
        set_energy_profile(EnergyProfile.ULTRA_PERFORMANCE)
        log.info("Energy profile set to ULTRA_PERFORMANCE")

    # 2. Warm up pools (force lazy initialization)
    io_pool = get_pool()
    try:
        cpu_pool = get_process_pool()
    except Exception:
        cpu_pool = None

    # 3. Pre-allocate buffers
    if prefill_buffers:
        _global_buffer_pool.buffer_size = buffer_size
        _global_buffer_pool.max_buffers = max_buffers
        # Pre-fill
        bufs = []
        for _ in range(min(max_buffers, 16)):
            bufs.append(_global_buffer_pool.acquire())
        for b in bufs:
            _global_buffer_pool.release(b)

    # 4. Build hardware manifest
    cpu_info = detect_cpu()
    gpus = detect_gpus()
    npus = detect_npus()
    ms = mem_stats()

    state.hardware_manifest = {
        "cpu": cpu_info.short_label(),
        "cpu_physical_cores": cpu_info.physical_cores,
        "cpu_logical_cores": cpu_info.logical_cores,
        "gpu_count": len(gpus),
        "gpu_usable": len([g for g in gpus if g.usable]),
        "gpu_best": gpus[0].short_label() if gpus else "N/A",
        "npu_count": len(npus),
        "npu_usable": len([n for n in npus if n.usable]),
        "ram_total_gb": round(ms.get("system_total_gb", 0), 1),
        "ram_available_gb": round(ms.get("system_available_gb", 0), 1),
        "memory_pressure": get_pressure().name,
        "io_pool_size": io_pool._max_workers,
        "cpu_pool_size": cpu_pool_size(),
        "buffers_pre_allocated": _global_buffer_pool.stats["pooled"],
    }

    state.active = True

    log.info(
        "MAX MODE ACTIVATED — CPU: %s | GPU: %d | NPU: %d | IO: %d threads | CPU: %d procs",
        cpu_info.short_label(),
        state.hardware_manifest["gpu_usable"],
        state.hardware_manifest["npu_usable"],
        state.hardware_manifest["io_pool_size"],
        state.hardware_manifest["cpu_pool_size"],
    )

    return state


def deactivate_max_mode(state: MaxModeState) -> Dict[str, Any]:
    """Restore previous OS settings and clean up resources.

    Parameters
    ----------
    state
        The ``MaxModeState`` returned by ``activate_max_mode()``.

    Returns
    -------
    dict
        Summary with elapsed time and resources used.
    """
    if not state.active:
        return {"error": "Max mode was not active"}

    elapsed = time.time() - state.activation_time

    # Restore OS settings
    set_task_priority(state.previous_priority)
    set_energy_profile(state.previous_energy)

    # Clean buffer pool
    _global_buffer_pool.clear()

    state.active = False

    summary = {
        "elapsed_s": round(elapsed, 2),
        "restored_priority": state.previous_priority.name,
        "restored_energy": state.previous_energy.name,
    }

    log.info("MAX MODE DEACTIVATED — ran for %.1f s", elapsed)
    return summary


# Global buffer pool for max mode
_global_buffer_pool = BufferPool(buffer_size=1 << 20, max_buffers=32)


# ═══════════════════════════════════════════════════════════════════════════
#  MaxMode Context Manager
# ═══════════════════════════════════════════════════════════════════════════

class MaxMode:
    """Context manager that activates maximum optimization mode.

    Automatically detects all hardware and configures the system for
    peak throughput on entry, and restores settings on exit.

    Usage::

        with MaxMode() as m:
            print(m.manifest)            # hardware manifest
            results = m.run_all(...)     # parallel execution
            result = m.run_io(fn, items) # I/O-bound parallel
            result = m.run_cpu(fn, items) # CPU-bound parallel

    Parameters
    ----------
    set_priority : bool
        Change OS scheduling priority (default True).
    set_energy : bool
        Change energy profile (default True).
    """

    def __init__(
        self,
        *,
        set_priority: bool = True,
        set_energy: bool = True,
    ):
        self._set_priority = set_priority
        self._set_energy = set_energy
        self._state: Optional[MaxModeState] = None

    def __enter__(self) -> "MaxMode":
        self._state = activate_max_mode(
            set_priority=self._set_priority,
            set_energy=self._set_energy,
        )
        return self

    def __exit__(self, *args: Any) -> None:
        if self._state:
            deactivate_max_mode(self._state)

    @property
    def active(self) -> bool:
        return self._state is not None and self._state.active

    @property
    def manifest(self) -> Dict[str, Any]:
        """Hardware manifest captured at activation."""
        if self._state:
            return self._state.hardware_manifest
        return {}

    @property
    def buffer_pool(self) -> BufferPool:
        """Shared buffer pool for I/O operations."""
        return _global_buffer_pool

    # ── Execution helpers ────────────────────────────────────────────────

    def run_io(
        self,
        fn: Callable[..., T],
        items: Sequence[tuple],
        *,
        max_concurrent: int = 0,
        desc: str = "I/O tasks",
        show_progress: bool = True,
    ) -> List[T]:
        """Run I/O-bound tasks in parallel using the full I/O thread pool.

        Uses the maximum available threads (io_pool_size) unless overridden.
        """
        if max_concurrent <= 0:
            max_concurrent = io_pool_size()
        return batch_execute(fn, items, max_concurrent, desc=desc, show_progress=show_progress)

    def run_cpu(
        self,
        fn: Callable[..., T],
        items: Sequence[tuple],
        *,
        max_workers: int = 0,
    ) -> List[T]:
        """Run CPU-bound tasks in parallel using the process pool.

        Distributes work across all physical cores.
        """
        if max_workers <= 0:
            max_workers = cpu_pool_size()
        pool = get_process_pool()
        futures: Dict[Future, int] = {}
        results: List[Optional[T]] = [None] * len(items)

        for idx, item in enumerate(items):
            futures[pool.submit(fn, *item)] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                log.warning("CPU task %d failed: %s", idx, exc)

        return results  # type: ignore[return-value]

    def run_all(
        self,
        *,
        cpu_fn: Optional[Callable[..., Any]] = None,
        cpu_items: Optional[Sequence[tuple]] = None,
        io_fn: Optional[Callable[..., Any]] = None,
        io_items: Optional[Sequence[tuple]] = None,
        gpu_fn: Optional[Callable[..., Any]] = None,
        gpu_items: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """Execute CPU, I/O, and GPU workloads simultaneously.

        All three workload types run in parallel:
          - CPU tasks → process pool (physical cores)
          - I/O tasks → thread pool (cores × 3)
          - GPU tasks → GPU dispatch (multi-GPU)

        Returns a dict with results for each workload type.

        Parameters
        ----------
        cpu_fn, cpu_items
            CPU-bound function and argument tuples.
        io_fn, io_items
            I/O-bound function and argument tuples.
        gpu_fn, gpu_items
            GPU function and data items.

        Returns
        -------
        dict
            ``{"cpu": [...], "io": [...], "gpu": [...], "elapsed_s": float}``
        """
        results: Dict[str, Any] = {}
        threads: List[threading.Thread] = []
        t0 = time.perf_counter()

        # CPU workload (in background thread that uses process pool)
        if cpu_fn and cpu_items:
            def _run_cpu() -> None:
                results["cpu"] = self.run_cpu(
                    cpu_fn, cpu_items,
                )
            t = threading.Thread(target=_run_cpu, name="maxmode-cpu")
            threads.append(t)
            t.start()

        # I/O workload (in background thread that uses I/O pool)
        if io_fn and io_items:
            def _run_io() -> None:
                results["io"] = self.run_io(
                    io_fn, io_items, show_progress=False,
                )
            t = threading.Thread(target=_run_io, name="maxmode-io")
            threads.append(t)
            t.start()

        # GPU workload (in background thread)
        if gpu_fn and gpu_items:
            def _run_gpu() -> None:
                from pyaccelerate.gpu.dispatch import dispatch
                gpus = detect_gpus()
                usable = [g for g in gpus if g.usable]
                if usable:
                    results["gpu"] = dispatch(gpu_fn, gpu_items, gpus=usable)
                else:
                    results["gpu"] = [gpu_fn(item) for item in gpu_items]
            t = threading.Thread(target=_run_gpu, name="maxmode-gpu")
            threads.append(t)
            t.start()

        # Wait for all
        for t in threads:
            t.join()

        results["elapsed_s"] = round(time.perf_counter() - t0, 4)

        log.info("MaxMode.run_all completed in %.3f s", results["elapsed_s"])
        return results

    def run_pipeline(
        self,
        stages: Sequence[tuple[str, Callable[..., Any], Sequence[tuple]]],
        *,
        stage_concurrency: Dict[str, int] | None = None,
    ) -> Dict[str, List[Any]]:
        """Run a multi-stage pipeline where each stage feeds the next.

        Parameters
        ----------
        stages
            List of ``(name, fn, items)`` tuples. Each stage runs in parallel.
        stage_concurrency
            Override concurrency per stage name.

        Returns
        -------
        dict
            ``{stage_name: [results], ...}``
        """
        all_results: Dict[str, List[Any]] = {}
        total_timer = Timer("pipeline")
        stage_concurrency = stage_concurrency or {}

        with total_timer:
            for name, fn, items in stages:
                conc = stage_concurrency.get(name, 0)
                with Timer(f"stage:{name}"):
                    stage_results = self.run_io(
                        fn, items,
                        max_concurrent=conc if conc > 0 else io_pool_size(),
                        desc=name,
                        show_progress=False,
                    )
                all_results[name] = stage_results
                log.info("Pipeline stage '%s': %d items completed", name, len(stage_results))

        log.info("Pipeline completed in %.3f s", total_timer.elapsed)
        return all_results

    def summary(self) -> str:
        """Human-readable summary of max mode state."""
        if not self._state or not self._state.active:
            return "MaxMode: INACTIVE"

        m = self._state.hardware_manifest
        elapsed = time.time() - self._state.activation_time
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║               MAX MODE — ACTIVE                            ║",
            "╠══════════════════════════════════════════════════════════════╣",
            f"║  CPU:     {m.get('cpu', 'N/A')}",
            f"║  GPU:     {m.get('gpu_usable', 0)} usable ({m.get('gpu_best', 'N/A')})",
            f"║  NPU:     {m.get('npu_usable', 0)} usable",
            f"║  RAM:     {m.get('ram_total_gb', 0)} GB total, {m.get('ram_available_gb', 0)} GB avail",
            f"║  IO Pool: {m.get('io_pool_size', 0)} threads",
            f"║  CPU Pool: {m.get('cpu_pool_size', 0)} processes",
            f"║  Buffers: {m.get('buffers_pre_allocated', 0)} pre-allocated",
            f"║  Pressure: {m.get('memory_pressure', 'N/A')}",
            f"║  Uptime:  {elapsed:.1f} s",
            "╚══════════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)
