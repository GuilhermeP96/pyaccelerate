"""
pyaccelerate.server — JSON HTTP & gRPC server for multi-language integration.

Exposes the full PyAccelerate API over HTTP (always available, stdlib-only)
and optionally over gRPC (when ``grpcio`` is installed).

HTTP API (all GET unless noted)::

    /api/v1/info        → engine.as_dict()
    /api/v1/summary     → engine.summary()
    /api/v1/cpu         → cpu.detect() as dict
    /api/v1/gpu         → gpu devices
    /api/v1/npu         → npu devices
    /api/v1/memory      → memory stats
    /api/v1/virt        → virtualization info
    /api/v1/benchmark   → run quick benchmark (POST for full)
    /api/v1/metrics     → prometheus metrics text
    /api/v1/status      → one-liner
    /api/v1/tune        → run auto-tune (POST)
    /api/v1/tune/profile→ current tune profile
    /health             → health check

gRPC service (port 50051 by default)::

    service PyAccelerate {
        rpc GetInfo (Empty) returns (InfoResponse);
        rpc GetSummary (Empty) returns (SummaryResponse);
        rpc GetCPU (Empty) returns (CPUResponse);
        rpc GetGPU (Empty) returns (GPUResponse);
        rpc GetNPU (Empty) returns (NPUResponse);
        rpc GetMemory (Empty) returns (MemoryResponse);
        rpc RunBenchmark (BenchmarkRequest) returns (BenchmarkResponse);
        rpc GetMetrics (Empty) returns (MetricsResponse);
    }

Usage::

    from pyaccelerate.server import PyAccelerateServer

    server = PyAccelerateServer()
    server.start(http_port=8420, grpc_port=50051)
    # ... server.stop()

CLI::

    pyaccelerate serve --http-port 8420 --grpc-port 50051
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

log = logging.getLogger("pyaccelerate.server")


# ═══════════════════════════════════════════════════════════════════════════
#  Lazy Engine singleton
# ═══════════════════════════════════════════════════════════════════════════

_engine_lock = threading.Lock()
_engine_instance: Any = None


def _get_engine() -> Any:
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                from pyaccelerate.engine import Engine
                _engine_instance = Engine()
    return _engine_instance


# ═══════════════════════════════════════════════════════════════════════════
#  Route handlers
# ═══════════════════════════════════════════════════════════════════════════

def _handle_info() -> Dict[str, Any]:
    return _get_engine().as_dict()


def _handle_summary() -> Dict[str, Any]:
    return {"summary": _get_engine().summary()}


def _handle_cpu() -> Dict[str, Any]:
    from pyaccelerate.cpu import detect
    c = detect()
    return {
        "brand": c.brand,
        "arch": c.arch,
        "physical_cores": c.physical_cores,
        "logical_cores": c.logical_cores,
        "frequency_mhz": c.frequency_mhz,
        "frequency_max_mhz": c.frequency_max_mhz,
        "numa_nodes": c.numa_nodes,
        "smt_ratio": c.smt_ratio,
        "is_arm": c.is_arm,
        "is_android": c.is_android,
        "is_sbc": c.is_sbc,
        "flags": c.flags,
    }


def _handle_gpu() -> Dict[str, Any]:
    from pyaccelerate.gpu import detect_all
    gpus = detect_all()
    return {
        "count": len(gpus),
        "devices": [g.as_dict() for g in gpus],
    }


def _handle_npu() -> Dict[str, Any]:
    from pyaccelerate.npu import detect_all
    npus = detect_all()
    return {
        "count": len(npus),
        "devices": [n.as_dict() for n in npus],
    }


def _handle_memory() -> Dict[str, Any]:
    from pyaccelerate.memory import get_stats, get_pressure
    return {
        "stats": get_stats(),
        "pressure": get_pressure().name,
    }


def _handle_virt() -> Dict[str, Any]:
    from pyaccelerate.virt import detect
    v = detect()
    return {
        "vtx": v.vtx_enabled,
        "hyperv": v.hyperv_running,
        "kvm": v.kvm_available,
        "wsl": v.wsl_available,
        "docker": v.docker_available,
        "inside_container": v.inside_container,
        "summary": v.summary_parts(),
    }


def _handle_benchmark(full: bool = False) -> Dict[str, Any]:
    from pyaccelerate.benchmark import run_all
    return run_all(quick=not full)


def _handle_metrics() -> str:
    from pyaccelerate.metrics import get_metrics_text
    return get_metrics_text()


def _handle_status() -> Dict[str, Any]:
    return {"status": _get_engine().status_line()}


def _handle_tune() -> Dict[str, Any]:
    from pyaccelerate.autotune import auto_tune
    profile = auto_tune(quick=True)
    from dataclasses import asdict
    return asdict(profile)


def _handle_tune_profile() -> Dict[str, Any]:
    from pyaccelerate.autotune import load_profile
    from dataclasses import asdict
    profile = load_profile()
    if profile is None:
        return {"profiled": False}
    return {**asdict(profile), "profiled": True}


# ═══════════════════════════════════════════════════════════════════════════
#  HTTP Server
# ═══════════════════════════════════════════════════════════════════════════

_ROUTES_GET: Dict[str, Any] = {
    "/api/v1/info": _handle_info,
    "/api/v1/summary": _handle_summary,
    "/api/v1/cpu": _handle_cpu,
    "/api/v1/gpu": _handle_gpu,
    "/api/v1/npu": _handle_npu,
    "/api/v1/memory": _handle_memory,
    "/api/v1/virt": _handle_virt,
    "/api/v1/status": _handle_status,
    "/api/v1/tune/profile": _handle_tune_profile,
}

_ROUTES_POST: Dict[str, Any] = {
    "/api/v1/benchmark": _handle_benchmark,
    "/api/v1/tune": _handle_tune,
}


class _APIHandler(BaseHTTPRequestHandler):
    """Handles JSON API requests."""

    server_version = "PyAccelerate-Server/1.0"

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, status: int = 200, content_type: str = "text/plain") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/health":
            self._send_text("OK")
            return

        if path == "/api/v1/metrics":
            self._send_text(_handle_metrics(), content_type="text/plain; version=0.0.4")
            return

        # Quick benchmark via GET
        if path == "/api/v1/benchmark":
            try:
                result = _handle_benchmark(full=False)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)
            return

        handler = _ROUTES_GET.get(path)
        if handler:
            try:
                result = handler()
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)
            return

        # API listing
        if path in ("/api", "/api/v1"):
            self._send_json({
                "endpoints": {
                    "GET": list(_ROUTES_GET.keys()) + ["/api/v1/benchmark", "/api/v1/metrics", "/health"],
                    "POST": list(_ROUTES_POST.keys()),
                },
            })
            return

        self._send_json({"error": "Not Found", "path": path}, 404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        handler = _ROUTES_POST.get(path)
        if handler:
            try:
                # Read body if present
                content_len = int(self.headers.get("Content-Length", 0))
                body = {}
                if content_len > 0:
                    raw = self.rfile.read(content_len)
                    body = json.loads(raw)

                # Pass `full` for benchmark
                if path == "/api/v1/benchmark":
                    full = body.get("full", False)
                    result = handler(full=full)
                else:
                    result = handler()

                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)
            return

        self._send_json({"error": "Not Found"}, 404)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.debug(format, *args)


# ═══════════════════════════════════════════════════════════════════════════
#  gRPC Server (optional — requires grpcio)
# ═══════════════════════════════════════════════════════════════════════════

def _start_grpc(port: int = 50051) -> Any:
    """Start gRPC server. Returns the server object or raises ImportError."""
    try:
        import grpc  # type: ignore[import-untyped]
        from concurrent.futures import ThreadPoolExecutor
    except ImportError:
        raise ImportError(
            "grpcio is required for gRPC server mode. "
            "Install it with: pip install pyaccelerate[grpc]"
        )

    from grpc import StatusCode  # type: ignore[import-untyped]

    class PyAccelerateServicer:
        """gRPC servicer using JSON serialization (no proto compilation needed)."""

        def GetInfo(self, request: Any, context: Any) -> Any:  # noqa: N802
            return _json_response(_handle_info())

        def GetSummary(self, request: Any, context: Any) -> Any:  # noqa: N802
            return _json_response(_handle_summary())

        def GetCPU(self, request: Any, context: Any) -> Any:  # noqa: N802
            return _json_response(_handle_cpu())

        def GetGPU(self, request: Any, context: Any) -> Any:  # noqa: N802
            return _json_response(_handle_gpu())

        def GetNPU(self, request: Any, context: Any) -> Any:  # noqa: N802
            return _json_response(_handle_npu())

        def GetMemory(self, request: Any, context: Any) -> Any:  # noqa: N802
            return _json_response(_handle_memory())

        def GetMetrics(self, request: Any, context: Any) -> Any:  # noqa: N802
            return _json_response({"metrics": _handle_metrics()})

        def RunBenchmark(self, request: Any, context: Any) -> Any:  # noqa: N802
            return _json_response(_handle_benchmark())

    def _json_response(data: Any) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    # Use generic RPC handlers (no .proto compilation needed)
    from grpc import unary_unary_rpc_method_handler, method_service_handler  # type: ignore

    # Simplified: use grpc.server with a basic servicer
    server = grpc.server(ThreadPoolExecutor(max_workers=4))

    # Register a generic handler that routes JSON-encoded requests
    service_name = "pyaccelerate.PyAccelerate"
    servicer = PyAccelerateServicer()

    method_map = {
        "GetInfo": servicer.GetInfo,
        "GetSummary": servicer.GetSummary,
        "GetCPU": servicer.GetCPU,
        "GetGPU": servicer.GetGPU,
        "GetNPU": servicer.GetNPU,
        "GetMemory": servicer.GetMemory,
        "GetMetrics": servicer.GetMetrics,
        "RunBenchmark": servicer.RunBenchmark,
    }

    class _GenericHandler(grpc.GenericRpcHandler):
        def service(self, handler_call_details: Any) -> Any:
            method = handler_call_details.method
            # method looks like "/pyaccelerate.PyAccelerate/GetInfo"
            if method:
                rpc_name = method.split("/")[-1]
                fn = method_map.get(rpc_name)
                if fn:
                    return grpc.unary_unary_rpc_method_handler(
                        lambda req, ctx, _fn=fn: _fn(req, ctx),
                        request_deserializer=lambda x: x,
                        response_serializer=lambda x: x,
                    )
            return None

    server.add_generic_rpc_handlers([_GenericHandler()])
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    log.info("gRPC server started → 0.0.0.0:%d", port)
    return server


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

class PyAccelerateServer:
    """Combined HTTP + gRPC server for multi-language integration.

    Parameters
    ----------
    http_port : int
        HTTP API port (default 8420).
    grpc_port : int
        gRPC port (default 50051).  Set to 0 to disable gRPC.
    host : str
        Bind address for HTTP (default ``0.0.0.0``).
    """

    def __init__(
        self,
        http_port: int = 8420,
        grpc_port: int = 50051,
        host: str = "0.0.0.0",
    ):
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.host = host
        self._http_server: Optional[HTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._grpc_server: Any = None

    def start(
        self,
        *,
        http: bool = True,
        grpc: bool = True,
        block: bool = False,
    ) -> None:
        """Start the server(s).

        Parameters
        ----------
        http : bool
            Start the HTTP API server.
        grpc : bool
            Start the gRPC server (if grpcio is installed).
        block : bool
            If *True*, block the calling thread until interrupted.
        """
        if http:
            self._start_http()
        if grpc and self.grpc_port > 0:
            try:
                self._grpc_server = _start_grpc(self.grpc_port)
            except ImportError as exc:
                log.warning("gRPC unavailable: %s", exc)

        if block:
            try:
                log.info("Server running. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                log.info("Shutting down…")
                self.stop()

    def _start_http(self) -> None:
        self._http_server = HTTPServer((self.host, self.http_port), _APIHandler)
        self._http_thread = threading.Thread(
            target=self._http_server.serve_forever,
            name="pyaccelerate-api",
            daemon=True,
        )
        self._http_thread.start()
        log.info("HTTP API server started → http://%s:%d/api/v1", self.host, self.http_port)

    def stop(self) -> None:
        """Stop all servers."""
        if self._http_server:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None
            self._http_thread = None
            log.info("HTTP server stopped")
        if self._grpc_server:
            self._grpc_server.stop(grace=2)
            self._grpc_server = None
            log.info("gRPC server stopped")

    @property
    def http_url(self) -> str:
        return f"http://{self.host}:{self.http_port}"

    @property
    def grpc_url(self) -> str:
        return f"{self.host}:{self.grpc_port}"

    def __enter__(self) -> "PyAccelerateServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
