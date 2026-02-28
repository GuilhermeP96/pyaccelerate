"""
pyaccelerate.engine — Unified orchestrator that auto-tunes all subsystems.

``Engine`` is the single entry-point for applications that want automatic
hardware detection & optimal configuration. It aggregates CPU, GPU, memory,
thread pool, and virtualization info into one coherent object.

Usage::

    from pyaccelerate import Engine

    engine = Engine()
    print(engine.summary())

    # Use the pre-configured thread pool
    engine.submit(my_func, arg1)

    # Run many tasks with auto-tuned concurrency
    engine.run_parallel(process, [(f,) for f in files])
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

from pyaccelerate.cpu import CPUInfo, detect as detect_cpu, recommend_workers
from pyaccelerate.gpu import GPUDevice, detect_all as detect_gpus, gpu_available, best_gpu, get_install_hint
from pyaccelerate.virt import VirtInfo, detect as detect_virt
from pyaccelerate.memory import Pressure, get_pressure, get_stats as mem_stats, clamp_workers
from pyaccelerate.threads import (
    get_pool,
    get_process_pool,
    run_parallel as _run_parallel,
    shutdown_pools,
    io_pool_size,
    submit as _submit,
    batch_execute,
)

log = logging.getLogger("pyaccelerate.engine")

T = TypeVar("T")


class Engine:
    """Unified high-performance engine — auto-detects and orchestrates all subsystems.

    Parameters
    ----------
    gpu_enabled : bool
        Enable GPU acceleration (auto-detected backends).
    multi_gpu : bool
        Distribute work across multiple GPUs when available.
    virt_enabled : bool
        Enable virtualization detection and integration.
    auto_threads : bool
        Auto-tune thread pool sizes based on hardware.
    max_io_workers : int
        Override I/O thread pool size (0 = auto).
    max_cpu_workers : int
        Override CPU process pool size (0 = auto).
    """

    def __init__(
        self,
        *,
        gpu_enabled: bool = True,
        multi_gpu: bool = True,
        virt_enabled: bool = True,
        auto_threads: bool = True,
        max_io_workers: int = 0,
        max_cpu_workers: int = 0,
    ):
        self.gpu_enabled = gpu_enabled
        self.multi_gpu = multi_gpu
        self.virt_enabled = virt_enabled
        self.auto_threads = auto_threads

        # Eager detection
        self._cpu: CPUInfo = detect_cpu()
        self._gpus: List[GPUDevice] = detect_gpus() if gpu_enabled else []
        self._virt: VirtInfo = detect_virt() if virt_enabled else VirtInfo()

        # Thread pool sizes
        if auto_threads:
            self._io_workers = max_io_workers or io_pool_size()
            self._cpu_workers = max_cpu_workers or self._cpu.physical_cores
        else:
            self._io_workers = max_io_workers or 8
            self._cpu_workers = max_cpu_workers or 4

        log.info(
            "Engine initialized: %s | GPU=%s | Virt=%s | IO=%d | CPU=%d",
            self._cpu.short_label(),
            self.best_gpu.short_label() if self.best_gpu else "N/A",
            ",".join(self._virt.summary_parts()) or "N/A",
            self._io_workers,
            self._cpu_workers,
        )

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def cpu(self) -> CPUInfo:
        return self._cpu

    @property
    def gpus(self) -> List[GPUDevice]:
        return self._gpus

    @property
    def usable_gpus(self) -> List[GPUDevice]:
        return [g for g in self._gpus if g.usable]

    @property
    def best_gpu(self) -> Optional[GPUDevice]:
        usable = self.usable_gpus
        return usable[0] if usable else None

    @property
    def virt(self) -> VirtInfo:
        return self._virt

    @property
    def memory_pressure(self) -> Pressure:
        return get_pressure()

    @property
    def io_workers(self) -> int:
        return self._io_workers

    @property
    def cpu_workers(self) -> int:
        return self._cpu_workers

    # ── Runtime toggles ─────────────────────────────────────────────────

    def set_gpu_enabled(self, on: bool) -> None:
        self.gpu_enabled = on
        if on and not self._gpus:
            self._gpus = detect_gpus()

    def set_multi_gpu(self, on: bool) -> None:
        self.multi_gpu = on

    # ── Task submission ──────────────────────────────────────────────────

    def submit(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> Future[T]:
        """Submit a single I/O-bound task to the virtual-thread pool."""
        return _submit(fn, *args, **kwargs)

    def run_parallel(
        self,
        fn: Callable[..., Any],
        items: Iterable[tuple],
        max_concurrent: int = 0,
        **kwargs: Any,
    ) -> int:
        """Run tasks with bounded concurrency on the virtual-thread pool.

        If *max_concurrent* is 0, uses ``recommend_workers(io_bound=True)``
        clamped by memory pressure.
        """
        if max_concurrent <= 0:
            max_concurrent = clamp_workers(recommend_workers(io_bound=True))
        return _run_parallel(fn, items, max_concurrent, **kwargs)

    def batch(
        self,
        fn: Callable[..., T],
        items: Sequence[tuple],
        max_concurrent: int = 0,
        desc: str = "Processing",
        show_progress: bool = True,
    ) -> List[T]:
        """Execute tasks with optional tqdm progress bar."""
        if max_concurrent <= 0:
            max_concurrent = clamp_workers(recommend_workers(io_bound=True))
        return batch_execute(fn, items, max_concurrent, desc=desc, show_progress=show_progress)

    def gpu_dispatch(
        self,
        fn: Callable[..., T],
        items: Sequence[Any],
        strategy: str = "round-robin",
    ) -> List[T]:
        """Dispatch work across GPUs (or fallback to CPU)."""
        from pyaccelerate.gpu.dispatch import dispatch
        gpus = self.usable_gpus if self.gpu_enabled else []
        return dispatch(fn, items, gpus=gpus or None, strategy=strategy)  # type: ignore[arg-type]

    # ── Shutdown ───────────────────────────────────────────────────────

    def shutdown(self, wait_for: bool = True) -> None:
        """Shut down all shared pools. Call during application exit."""
        shutdown_pools(wait_for)

    # ── Info ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable multi-line report of all subsystems."""
        lines: list[str] = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║              PyAccelerate — Engine Report                   ║",
            "╠══════════════════════════════════════════════════════════════╣",
        ]

        # CPU
        c = self._cpu
        lines.append(f"║  CPU: {c.short_label()}")
        if c.frequency_max_mhz:
            lines.append(f"║       Base: {c.frequency_mhz:.0f} MHz | Boost: {c.frequency_max_mhz:.0f} MHz")
        if c.numa_nodes > 1:
            lines.append(f"║       NUMA: {c.numa_nodes} nodes")
        if c.flags:
            lines.append(f"║       ISA: {', '.join(c.flags[:8])}")

        # GPU
        usable = self.usable_gpus
        if usable and self.gpu_enabled:
            best = usable[0]
            lines.append(f"║  GPU: {best.short_label()}")
            if len(usable) > 1:
                mode = "multi-GPU ACTIVE" if self.multi_gpu else "available"
                others = ", ".join(g.name for g in usable[1:])
                lines.append(f"║       + {others} ({mode})")
        elif usable:
            lines.append(f"║  GPU: {usable[0].name} (DISABLED by user)")
        else:
            all_g = self._gpus
            if all_g:
                lines.append(f"║  GPU: {all_g[0].name} (no compute library)")
                hint = get_install_hint()
                if hint:
                    lines.append(f"║       Hint: {hint}")
            else:
                lines.append("║  GPU: None detected")

        # Memory
        ms = mem_stats()
        if "system_total_gb" in ms:
            lines.append(
                f"║  RAM: {ms['system_total_gb']:.1f} GB total | "
                f"{ms['system_available_gb']:.1f} GB available | "
                f"Pressure: {self.memory_pressure.name}"
            )

        # Virtualization
        vp = self._virt.summary_parts()
        if vp:
            lines.append(f"║  Virt: {', '.join(vp)}")
        else:
            lines.append("║  Virt: None detected")

        # Pools
        lines.append(
            f"║  Pools: IO={self._io_workers} virtual threads | "
            f"CPU={self._cpu_workers} processes"
        )

        lines.append("╚══════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    def status_line(self) -> str:
        """One-line summary for status bars / headers."""
        parts: list[str] = []

        # CPU
        parts.append(f"CPU: {self._cpu.logical_cores}T")

        # GPU
        usable = self.usable_gpus
        if usable and self.gpu_enabled:
            best = usable[0]
            txt = f"GPU: {best.name}"
            if self.multi_gpu and len(usable) > 1:
                txt += f" +{len(usable) - 1}"
            parts.append(txt)
        else:
            parts.append("GPU: N/A")

        # Memory pressure
        parts.append(f"MEM: {self.memory_pressure.name}")

        # Pool
        parts.append(f"VT: {self._io_workers}")

        return "  |  ".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        """Machine-readable snapshot of engine state."""
        return {
            "cpu": {
                "brand": self._cpu.brand,
                "arch": self._cpu.arch,
                "physical_cores": self._cpu.physical_cores,
                "logical_cores": self._cpu.logical_cores,
                "freq_mhz": self._cpu.frequency_mhz,
                "freq_max_mhz": self._cpu.frequency_max_mhz,
                "numa_nodes": self._cpu.numa_nodes,
                "flags": self._cpu.flags,
            },
            "gpu": {
                "enabled": self.gpu_enabled,
                "multi_gpu": self.multi_gpu,
                "devices": [g.as_dict() for g in self._gpus],
                "best": self.best_gpu.as_dict() if self.best_gpu else None,
            },
            "memory": mem_stats(),
            "memory_pressure": self.memory_pressure.name,
            "virt": {
                "vtx": self._virt.vtx_enabled,
                "hyperv": self._virt.hyperv_running,
                "kvm": self._virt.kvm_available,
                "wsl": self._virt.wsl_available,
                "docker": self._virt.docker_available,
                "inside_container": self._virt.inside_container,
            },
            "pools": {
                "io_workers": self._io_workers,
                "cpu_workers": self._cpu_workers,
            },
        }
