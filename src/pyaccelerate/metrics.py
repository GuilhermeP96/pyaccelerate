"""
pyaccelerate.metrics — Prometheus metrics exporter.

Exposes CPU, GPU, NPU, memory, thread-pool, and virtualization metrics in
the `Prometheus exposition format`_ so they can be scraped by any
Prometheus-compatible collector (Grafana Agent, VictoriaMetrics, etc.).

A lightweight HTTP server (stdlib only — no Flask/FastAPI needed) is
included for standalone operation.

Usage::

    from pyaccelerate.metrics import start_metrics_server, get_metrics_text

    # Start the /metrics endpoint on port 9090
    server = start_metrics_server(port=9090)

    # Or just get the text for your own framework
    text = get_metrics_text()

.. _Prometheus exposition format:
   https://prometheus.io/docs/instrumenting/exposition_formats/
"""

from __future__ import annotations

import logging
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("pyaccelerate.metrics")

# ── Metric types ──────────────────────────────────────────────────────────

_METRIC_PREFIX = "pyaccelerate"


def _gauge(name: str, help_text: str, value: float, labels: Dict[str, str] | None = None) -> str:
    """Format a single Prometheus gauge metric."""
    fqn = f"{_METRIC_PREFIX}_{name}"
    label_str = ""
    if labels:
        pairs = ",".join(f'{k}="{v}"' for k, v in labels.items())
        label_str = f"{{{pairs}}}"
    return (
        f"# HELP {fqn} {help_text}\n"
        f"# TYPE {fqn} gauge\n"
        f"{fqn}{label_str} {value}\n"
    )


def _gauge_multi(
    name: str,
    help_text: str,
    samples: List[Tuple[Dict[str, str], float]],
) -> str:
    """Format a gauge with multiple label sets."""
    fqn = f"{_METRIC_PREFIX}_{name}"
    lines = [
        f"# HELP {fqn} {help_text}",
        f"# TYPE {fqn} gauge",
    ]
    for labels, value in samples:
        pairs = ",".join(f'{k}="{v}"' for k, v in labels.items())
        lines.append(f"{fqn}{{{pairs}}} {value}")
    return "\n".join(lines) + "\n"


# ── Collectors ────────────────────────────────────────────────────────────

def _collect_cpu() -> str:
    """CPU metrics."""
    try:
        from pyaccelerate.cpu import detect
        c = detect()
    except Exception:
        return ""

    parts: list[str] = []
    parts.append(_gauge("cpu_physical_cores", "Number of physical CPU cores", c.physical_cores))
    parts.append(_gauge("cpu_logical_cores", "Number of logical CPU cores (threads)", c.logical_cores))
    parts.append(_gauge("cpu_frequency_mhz", "CPU base frequency in MHz",
                        c.frequency_mhz, {"type": "base"}))
    if c.frequency_max_mhz:
        parts.append(_gauge("cpu_frequency_mhz", "CPU frequency in MHz",
                            c.frequency_max_mhz, {"type": "max"}))
    parts.append(_gauge("cpu_numa_nodes", "Number of NUMA nodes", c.numa_nodes))
    parts.append(_gauge("cpu_is_arm", "1 if running on ARM architecture", int(c.is_arm)))
    parts.append(_gauge("cpu_is_sbc", "1 if running on a single-board computer", int(c.is_sbc)))
    return "".join(parts)


def _collect_memory() -> str:
    """Memory metrics."""
    try:
        from pyaccelerate.memory import get_stats, get_pressure, Pressure
        stats = get_stats()
        pressure = get_pressure()
    except Exception:
        return ""

    parts: list[str] = []
    total_b = stats.get("system_total_gb", 0) * (1024 ** 3)
    avail_b = stats.get("system_available_gb", 0) * (1024 ** 3)
    parts.append(_gauge("memory_total_bytes", "Total system memory in bytes", total_b))
    parts.append(_gauge("memory_available_bytes", "Available system memory in bytes", avail_b))

    # Pressure as 0-3 ordinal (LOW=0, MODERATE=1, HIGH=2, CRITICAL=3)
    pressure_map = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
    pval = pressure_map.get(pressure.name, 0)
    parts.append(_gauge("memory_pressure", "Memory pressure level (0=low, 3=critical)", pval))

    proc_rss = stats.get("process_rss_gb", 0) * (1024 ** 3)
    parts.append(_gauge("memory_process_rss_bytes", "Process RSS in bytes", proc_rss))
    return "".join(parts)


def _collect_gpu() -> str:
    """GPU metrics."""
    try:
        from pyaccelerate.gpu import detect_all
        gpus = detect_all()
    except Exception:
        return ""

    parts: list[str] = []
    parts.append(_gauge("gpu_count", "Number of detected GPUs", len(gpus)))

    if not gpus:
        return "".join(parts)

    samples_mem: list[tuple[dict[str, str], float]] = []
    samples_score: list[tuple[dict[str, str], float]] = []
    samples_usable: list[tuple[dict[str, str], float]] = []
    samples_cu: list[tuple[dict[str, str], float]] = []

    for i, g in enumerate(gpus):
        lbl = {"device": g.name, "index": str(i), "backend": g.backend, "vendor": g.vendor}
        samples_mem.append((lbl, g.memory_gb * (1024 ** 3)))
        samples_score.append((lbl, g.score))
        samples_usable.append((lbl, int(g.usable)))
        if g.compute_units:
            samples_cu.append((lbl, g.compute_units))

    parts.append(_gauge_multi("gpu_memory_bytes", "GPU memory in bytes", samples_mem))
    parts.append(_gauge_multi("gpu_score", "GPU compute score", samples_score))
    parts.append(_gauge_multi("gpu_usable", "1 if GPU is usable for compute", samples_usable))
    if samples_cu:
        parts.append(_gauge_multi("gpu_compute_units", "GPU compute units", samples_cu))
    return "".join(parts)


def _collect_npu() -> str:
    """NPU metrics."""
    try:
        from pyaccelerate.npu import detect_all
        npus = detect_all()
    except Exception:
        return ""

    parts: list[str] = []
    parts.append(_gauge("npu_count", "Number of detected NPUs", len(npus)))

    if not npus:
        return "".join(parts)

    samples_tops: list[tuple[dict[str, str], float]] = []
    samples_score: list[tuple[dict[str, str], float]] = []
    samples_usable: list[tuple[dict[str, str], float]] = []

    for i, n in enumerate(npus):
        lbl = {"device": n.name, "index": str(i), "backend": n.backend, "vendor": n.vendor}
        samples_tops.append((lbl, n.tops))
        samples_score.append((lbl, n.score))
        samples_usable.append((lbl, int(n.usable)))

    parts.append(_gauge_multi("npu_tops", "NPU throughput in TOPS", samples_tops))
    parts.append(_gauge_multi("npu_score", "NPU compute score", samples_score))
    parts.append(_gauge_multi("npu_usable", "1 if NPU is usable for inference", samples_usable))
    return "".join(parts)


def _collect_threads() -> str:
    """Thread pool metrics."""
    try:
        from pyaccelerate.threads import io_pool_size
        from pyaccelerate.cpu import recommend_workers
    except Exception:
        return ""

    parts: list[str] = []
    parts.append(_gauge("threadpool_io_size", "I/O virtual thread pool size", io_pool_size()))
    parts.append(
        _gauge("threadpool_recommended_io_workers",
               "Recommended I/O workers", recommend_workers(io_bound=True))
    )
    parts.append(
        _gauge("threadpool_recommended_cpu_workers",
               "Recommended CPU workers", recommend_workers(io_bound=False))
    )
    return "".join(parts)


def _collect_virt() -> str:
    """Virtualization metrics."""
    try:
        from pyaccelerate.virt import detect
        v = detect()
    except Exception:
        return ""

    parts: list[str] = []
    parts.append(_gauge("virt_in_container", "1 if running inside a container", int(v.inside_container)))
    parts.append(_gauge("virt_docker", "1 if Docker is available", int(v.docker_available)))
    parts.append(_gauge("virt_wsl", "1 if WSL is available", int(v.wsl_available)))
    parts.append(_gauge("virt_hyperv", "1 if Hyper-V is running", int(v.hyperv_running)))
    parts.append(_gauge("virt_kvm", "1 if KVM is available", int(v.kvm_available)))
    return "".join(parts)


def _collect_autotune() -> str:
    """Auto-tune profile metrics (if a profile exists)."""
    try:
        from pyaccelerate.autotune import load_profile
        profile = load_profile()
    except Exception:
        return ""

    if profile is None:
        return _gauge("autotune_profiled", "1 if an auto-tune profile exists", 0)

    parts: list[str] = []
    parts.append(_gauge("autotune_profiled", "1 if an auto-tune profile exists", 1))
    parts.append(_gauge("autotune_overall_score", "Overall auto-tune score (0-100)", profile.overall_score))
    parts.append(_gauge("autotune_cpu_score", "CPU auto-tune score (0-100)", profile.cpu_score))
    parts.append(_gauge("autotune_memory_score", "Memory auto-tune score (0-100)", profile.memory_score))
    parts.append(_gauge("autotune_gpu_score", "GPU auto-tune score (0-100)", profile.gpu_score))
    parts.append(
        _gauge("autotune_optimal_io_workers", "Optimal I/O workers from tune", profile.optimal_io_workers)
    )
    parts.append(
        _gauge("autotune_optimal_cpu_workers", "Optimal CPU workers from tune", profile.optimal_cpu_workers)
    )
    return "".join(parts)


# ── Public API ────────────────────────────────────────────────────────────

def get_metrics_text() -> str:
    """Return all metrics as a Prometheus-format string."""
    sections = [
        _collect_cpu(),
        _collect_memory(),
        _collect_gpu(),
        _collect_npu(),
        _collect_threads(),
        _collect_virt(),
        _collect_autotune(),
    ]
    return "\n".join(s for s in sections if s)


# ── HTTP server ───────────────────────────────────────────────────────────

class _MetricsHandler(BaseHTTPRequestHandler):
    """Serves /metrics in Prometheus exposition format."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/metrics":
            body = get_metrics_text().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.debug(format, *args)


_server_instance: Optional[HTTPServer] = None
_server_thread: Optional[threading.Thread] = None


def start_metrics_server(
    port: int = 9090,
    host: str = "0.0.0.0",
) -> HTTPServer:
    """Start a background HTTP server exposing ``/metrics``.

    Parameters
    ----------
    port : int
        TCP port to listen on (default 9090).
    host : str
        Bind address (default ``0.0.0.0``).

    Returns
    -------
    HTTPServer
        The running server instance. Call :func:`stop_metrics_server` to
        shut it down.
    """
    global _server_instance, _server_thread

    if _server_instance is not None:
        log.warning("Metrics server already running on %s:%d", host, port)
        return _server_instance

    _server_instance = HTTPServer((host, port), _MetricsHandler)
    _server_thread = threading.Thread(
        target=_server_instance.serve_forever,
        name="pyaccelerate-metrics",
        daemon=True,
    )
    _server_thread.start()
    log.info("Prometheus metrics server started → http://%s:%d/metrics", host, port)
    return _server_instance


def stop_metrics_server() -> None:
    """Stop a running metrics server."""
    global _server_instance, _server_thread
    if _server_instance is not None:
        _server_instance.shutdown()
        _server_instance.server_close()
        _server_instance = None
        _server_thread = None
        log.info("Metrics server stopped")
