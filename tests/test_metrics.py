"""Tests for pyaccelerate.metrics — Prometheus metrics exporter."""

from __future__ import annotations

import re
import socket
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from pyaccelerate.metrics import (
    get_metrics_text,
    _gauge,
    _gauge_multi,
    _collect_cpu,
    _collect_memory,
    _collect_gpu,
    _collect_npu,
    _collect_threads,
    _collect_virt,
    _collect_autotune,
    start_metrics_server,
    stop_metrics_server,
)


# ── Gauge formatting ─────────────────────────────────────────────────────

class TestGaugeFormat:
    def test_simple_gauge(self):
        text = _gauge("test_metric", "A test metric", 42.5)
        assert "# HELP pyaccelerate_test_metric A test metric" in text
        assert "# TYPE pyaccelerate_test_metric gauge" in text
        assert "pyaccelerate_test_metric 42.5" in text

    def test_gauge_with_labels(self):
        text = _gauge("cpu_freq", "CPU frequency", 3500, {"type": "max"})
        assert 'pyaccelerate_cpu_freq{type="max"} 3500' in text

    def test_gauge_multi(self):
        samples = [
            ({"device": "GPU-0"}, 8.0),
            ({"device": "GPU-1"}, 16.0),
        ]
        text = _gauge_multi("gpu_mem", "GPU memory", samples)
        assert "# HELP pyaccelerate_gpu_mem" in text
        assert '# TYPE pyaccelerate_gpu_mem gauge' in text
        assert 'device="GPU-0"' in text
        assert 'device="GPU-1"' in text

    def test_gauge_multi_empty(self):
        text = _gauge_multi("empty", "No data", [])
        assert "# TYPE" in text
        # No samples, just header


# ── Collectors ────────────────────────────────────────────────────────────

class TestCollectors:
    def test_collect_cpu_returns_string(self):
        text = _collect_cpu()
        assert isinstance(text, str)
        assert "pyaccelerate_cpu_physical_cores" in text
        assert "pyaccelerate_cpu_logical_cores" in text

    def test_collect_memory_returns_string(self):
        text = _collect_memory()
        assert isinstance(text, str)
        assert "pyaccelerate_memory_total_bytes" in text
        assert "pyaccelerate_memory_pressure" in text

    def test_collect_gpu_returns_string(self):
        text = _collect_gpu()
        assert isinstance(text, str)
        assert "pyaccelerate_gpu_count" in text

    def test_collect_npu_returns_string(self):
        text = _collect_npu()
        assert isinstance(text, str)
        assert "pyaccelerate_npu_count" in text

    def test_collect_threads_returns_string(self):
        text = _collect_threads()
        assert isinstance(text, str)
        assert "pyaccelerate_threadpool" in text

    def test_collect_virt_returns_string(self):
        text = _collect_virt()
        assert isinstance(text, str)
        assert "pyaccelerate_virt" in text

    def test_collect_autotune_no_profile(self):
        text = _collect_autotune()
        assert isinstance(text, str)
        assert "pyaccelerate_autotune_profiled" in text


# ── Full metrics ──────────────────────────────────────────────────────────

class TestGetMetricsText:
    def test_returns_string(self):
        text = get_metrics_text()
        assert isinstance(text, str)
        assert len(text) > 100

    def test_contains_all_sections(self):
        text = get_metrics_text()
        assert "pyaccelerate_cpu" in text
        assert "pyaccelerate_memory" in text
        assert "pyaccelerate_gpu" in text
        assert "pyaccelerate_npu" in text
        assert "pyaccelerate_threadpool" in text
        assert "pyaccelerate_virt" in text

    def test_prometheus_format(self):
        text = get_metrics_text()
        lines = text.strip().split("\n")
        for line in lines:
            if line.startswith("#"):
                assert line.startswith("# HELP") or line.startswith("# TYPE")
            elif line.strip():
                # metric line: name{labels} value
                assert re.match(r"pyaccelerate_\w+", line), f"Bad line: {line}"


# ── HTTP Server ───────────────────────────────────────────────────────────

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestMetricsServer:
    def test_start_stop(self):
        port = _find_free_port()
        server = start_metrics_server(port=port, host="127.0.0.1")
        assert server is not None
        time.sleep(0.3)

        # Request /metrics
        import urllib.request
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)
            body = resp.read().decode()
            assert "pyaccelerate_cpu" in body
            assert resp.status == 200
        finally:
            stop_metrics_server()

    def test_health_endpoint(self):
        port = _find_free_port()
        start_metrics_server(port=port, host="127.0.0.1")
        time.sleep(0.3)

        import urllib.request
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5)
            assert resp.read() == b"OK"
        finally:
            stop_metrics_server()

    def test_404_unknown_path(self):
        port = _find_free_port()
        start_metrics_server(port=port, host="127.0.0.1")
        time.sleep(0.3)

        import urllib.request, urllib.error
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/unknown", timeout=5)
            assert False, "Should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 404
        finally:
            stop_metrics_server()
