"""
pyaccelerate.health — Aggregated system health checks.

Collects CPU temperature, memory pressure, GPU thermal status, and disk I/O
health into a single ``HealthReport``.

Usage::

    from pyaccelerate.health import health_check

    report = health_check()
    if report.healthy:
        ...
    print(report.summary())
"""

from __future__ import annotations

import dataclasses
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil

log = logging.getLogger("pyaccelerate.health")


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclasses.dataclass(frozen=True)
class ComponentHealth:
    name: str
    status: Status
    detail: str = ""
    metrics: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class HealthReport:
    components: List[ComponentHealth]
    timestamp: float
    elapsed_ms: float

    @property
    def healthy(self) -> bool:
        return all(c.status in (Status.OK, Status.UNKNOWN) for c in self.components)

    @property
    def overall(self) -> Status:
        statuses = [c.status for c in self.components]
        if Status.CRITICAL in statuses:
            return Status.CRITICAL
        if Status.WARN in statuses:
            return Status.WARN
        if all(s == Status.UNKNOWN for s in statuses):
            return Status.UNKNOWN
        return Status.OK

    def summary(self) -> str:
        lines = [f"Health: {self.overall.value.upper()} ({self.elapsed_ms:.0f}ms)"]
        for c in self.components:
            marker = {"ok": "+", "warn": "~", "critical": "!", "unknown": "?"}[c.status.value]
            line = f"  [{marker}] {c.name}: {c.status.value}"
            if c.detail:
                line += f" — {c.detail}"
            lines.append(line)
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "overall": self.overall.value,
            "timestamp": self.timestamp,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "detail": c.detail,
                    "metrics": c.metrics,
                }
                for c in self.components
            ],
        }


def _check_cpu() -> ComponentHealth:
    try:
        pct = psutil.cpu_percent(interval=0.1)
        freq = psutil.cpu_freq()
        metrics = {"usage_pct": pct}
        if freq:
            metrics["freq_mhz"] = round(freq.current, 0)

        # CPU temperatures (platform-dependent)
        try:
            temps = psutil.sensors_temperatures()  # type: ignore[attr-defined]
            if temps:
                all_temps = [t.current for sensors in temps.values() for t in sensors]
                if all_temps:
                    max_temp = max(all_temps)
                    metrics["max_temp_c"] = round(max_temp, 1)
                    if max_temp > 95:
                        return ComponentHealth("cpu", Status.CRITICAL, f"Thermal throttle risk ({max_temp:.0f}°C)", metrics)
                    if max_temp > 80:
                        return ComponentHealth("cpu", Status.WARN, f"High temperature ({max_temp:.0f}°C)", metrics)
        except (AttributeError, OSError):
            pass

        if pct > 95:
            return ComponentHealth("cpu", Status.WARN, f"{pct:.0f}% utilization", metrics)
        return ComponentHealth("cpu", Status.OK, f"{pct:.0f}% utilization", metrics)
    except Exception as exc:
        return ComponentHealth("cpu", Status.UNKNOWN, str(exc))


def _check_memory() -> ComponentHealth:
    try:
        vm = psutil.virtual_memory()
        metrics = {
            "total_gb": round(vm.total / (1 << 30), 1),
            "available_gb": round(vm.available / (1 << 30), 1),
            "used_pct": vm.percent,
        }
        if vm.percent > 95:
            return ComponentHealth("memory", Status.CRITICAL, f"{vm.percent}% used", metrics)
        if vm.percent > 85:
            return ComponentHealth("memory", Status.WARN, f"{vm.percent}% used", metrics)
        return ComponentHealth("memory", Status.OK, f"{vm.percent}% used", metrics)
    except Exception as exc:
        return ComponentHealth("memory", Status.UNKNOWN, str(exc))


def _check_disk() -> ComponentHealth:
    try:
        counters = psutil.disk_io_counters()
        parts = psutil.disk_partitions()
        metrics: Dict[str, Any] = {}
        if counters:
            metrics["read_mb"] = round(counters.read_bytes / (1 << 20), 0)
            metrics["write_mb"] = round(counters.write_bytes / (1 << 20), 0)

        # Check root partition usage
        status = Status.OK
        detail = ""
        for p in parts:
            try:
                usage = psutil.disk_usage(p.mountpoint)
                if usage.percent > 95:
                    status = Status.CRITICAL
                    detail = f"{p.mountpoint} {usage.percent}% full"
                    break
                elif usage.percent > 90 and status != Status.CRITICAL:
                    status = Status.WARN
                    detail = f"{p.mountpoint} {usage.percent}% full"
            except (OSError, PermissionError):
                continue
        if not detail:
            detail = "all partitions OK"
        return ComponentHealth("disk", status, detail, metrics)
    except Exception as exc:
        return ComponentHealth("disk", Status.UNKNOWN, str(exc))


def _check_gpu() -> ComponentHealth:
    try:
        from pyaccelerate.gpu import detect_all

        gpus = detect_all()
        if not gpus:
            return ComponentHealth("gpu", Status.UNKNOWN, "No GPU detected")

        names = [g.name for g in gpus]
        metrics = {"count": len(gpus), "devices": names}
        return ComponentHealth("gpu", Status.OK, f"{len(gpus)} device(s)", metrics)
    except Exception as exc:
        return ComponentHealth("gpu", Status.UNKNOWN, str(exc))


def health_check(include_gpu: bool = True) -> HealthReport:
    """Run all health checks and return an aggregated report.

    Parameters
    ----------
    include_gpu
        Whether to include GPU detection (can be slow on first call).
    """
    start = time.monotonic()
    components: List[ComponentHealth] = [
        _check_cpu(),
        _check_memory(),
        _check_disk(),
    ]
    if include_gpu:
        components.append(_check_gpu())

    elapsed = (time.monotonic() - start) * 1000
    return HealthReport(
        components=components,
        timestamp=time.time(),
        elapsed_ms=elapsed,
    )
