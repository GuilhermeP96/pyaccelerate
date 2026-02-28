"""Tests for pyaccelerate.server — JSON HTTP & gRPC server."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
import urllib.error
from unittest.mock import patch

import pytest

from pyaccelerate.server import PyAccelerateServer, _handle_info, _handle_cpu


# ── Helpers ───────────────────────────────────────────────────────────────

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(port: int, path: str) -> tuple[int, dict | str]:
    """GET request, return (status, parsed body)."""
    url = f"http://127.0.0.1:{port}{path}"
    try:
        resp = urllib.request.urlopen(url, timeout=10)
        body = resp.read().decode()
        ct = resp.headers.get("Content-Type", "")
        if "json" in ct:
            return resp.status, json.loads(body)
        return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body


def _post(port: int, path: str, data: dict | None = None) -> tuple[int, dict | str]:
    """POST request, return (status, parsed body)."""
    url = f"http://127.0.0.1:{port}{path}"
    payload = json.dumps(data or {}).encode()
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        body = resp.read().decode()
        ct = resp.headers.get("Content-Type", "")
        if "json" in ct:
            return resp.status, json.loads(body)
        return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body


# ── Handler functions ─────────────────────────────────────────────────────

class TestHandlers:
    def test_handle_info(self):
        result = _handle_info()
        assert "cpu" in result
        assert "gpu" in result
        assert "memory" in result

    def test_handle_cpu(self):
        result = _handle_cpu()
        assert "brand" in result
        assert result["physical_cores"] >= 1


# ── Server lifecycle ──────────────────────────────────────────────────────

class TestServerLifecycle:
    def test_context_manager(self):
        port = _find_free_port()
        with PyAccelerateServer(http_port=port, grpc_port=0, host="127.0.0.1") as srv:
            time.sleep(0.3)
            status, body = _get(port, "/health")
            assert status == 200

    def test_start_stop(self):
        port = _find_free_port()
        srv = PyAccelerateServer(http_port=port, grpc_port=0, host="127.0.0.1")
        srv.start(http=True, grpc=False)
        time.sleep(0.3)
        status, _ = _get(port, "/health")
        assert status == 200
        srv.stop()

    def test_urls(self):
        srv = PyAccelerateServer(http_port=8420, grpc_port=50051)
        assert "8420" in srv.http_url
        assert "50051" in srv.grpc_url


# ── HTTP API Endpoints ────────────────────────────────────────────────────

class TestHTTPEndpoints:
    @pytest.fixture(autouse=True)
    def _server(self):
        self.port = _find_free_port()
        self.srv = PyAccelerateServer(http_port=self.port, grpc_port=0, host="127.0.0.1")
        self.srv.start(http=True, grpc=False)
        time.sleep(0.3)
        yield
        self.srv.stop()

    def test_health(self):
        status, body = _get(self.port, "/health")
        assert status == 200
        assert body == "OK"

    def test_api_listing(self):
        status, body = _get(self.port, "/api/v1")
        assert status == 200
        assert "endpoints" in body
        assert "GET" in body["endpoints"]

    def test_info(self):
        status, body = _get(self.port, "/api/v1/info")
        assert status == 200
        assert "cpu" in body
        assert "gpu" in body

    def test_summary(self):
        status, body = _get(self.port, "/api/v1/summary")
        assert status == 200
        assert "summary" in body
        assert "PyAccelerate" in body["summary"]

    def test_cpu(self):
        status, body = _get(self.port, "/api/v1/cpu")
        assert status == 200
        assert body["physical_cores"] >= 1

    def test_gpu(self):
        status, body = _get(self.port, "/api/v1/gpu")
        assert status == 200
        assert "count" in body
        assert "devices" in body

    def test_npu(self):
        status, body = _get(self.port, "/api/v1/npu")
        assert status == 200
        assert "count" in body

    def test_memory(self):
        status, body = _get(self.port, "/api/v1/memory")
        assert status == 200
        assert "pressure" in body

    def test_virt(self):
        status, body = _get(self.port, "/api/v1/virt")
        assert status == 200
        assert "inside_container" in body

    def test_status(self):
        status, body = _get(self.port, "/api/v1/status")
        assert status == 200
        assert "status" in body

    def test_metrics(self):
        status, body = _get(self.port, "/api/v1/metrics")
        assert status == 200
        assert "pyaccelerate_cpu" in body

    def test_tune_profile_none(self):
        status, body = _get(self.port, "/api/v1/tune/profile")
        assert status == 200
        assert body.get("profiled") is False

    def test_404(self):
        status, body = _get(self.port, "/api/v1/nonexistent")
        assert status == 404

    def test_cors_headers(self):
        url = f"http://127.0.0.1:{self.port}/api/v1/status"
        resp = urllib.request.urlopen(url, timeout=5)
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
