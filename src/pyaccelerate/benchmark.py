"""
pyaccelerate.benchmark — Built-in micro-benchmarks for the current host.

Runs quick benchmarks to characterise host performance:
  - CPU single-thread & multi-thread throughput
  - Memory bandwidth
  - I/O thread pool latency
  - GPU compute throughput (if available)

Results are returned as structured dicts suitable for logging, dashboards
or automated tuning decisions.

Usage::

    from pyaccelerate.benchmark import run_all, run_cpu, run_gpu

    report = run_all()
    print(report)
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

log = logging.getLogger("pyaccelerate.benchmark")


# ═══════════════════════════════════════════════════════════════════════════
#  CPU benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def run_cpu(
    iterations: int = 500_000,
    hash_bytes: int = 4096,
) -> Dict[str, Any]:
    """Single-thread CPU throughput:  math + hashing workload."""
    # Math workload
    t0 = time.perf_counter()
    total = 0.0
    for i in range(1, iterations + 1):
        total += math.sqrt(i) * math.sin(i)
    math_time = time.perf_counter() - t0

    # Hash workload
    data = os.urandom(hash_bytes)
    t0 = time.perf_counter()
    for _ in range(iterations):
        hashlib.md5(data).hexdigest()
    hash_time = time.perf_counter() - t0

    return {
        "benchmark": "cpu_single_thread",
        "iterations": iterations,
        "math_time_s": round(math_time, 4),
        "hash_time_s": round(hash_time, 4),
        "total_s": round(math_time + hash_time, 4),
        "math_ops_per_sec": int(iterations / math_time) if math_time > 0 else 0,
        "hash_ops_per_sec": int(iterations / hash_time) if hash_time > 0 else 0,
    }


def run_cpu_multithread(
    iterations: int = 200_000,
    workers: int = 0,
) -> Dict[str, Any]:
    """Multi-thread CPU throughput using standard thread pool."""
    if workers <= 0:
        workers = os.cpu_count() or 4

    def _work(start: int, end: int) -> float:
        total = 0.0
        for i in range(start, end):
            total += math.sqrt(i) * math.sin(i)
        return total

    chunk = iterations // workers
    ranges = [(i * chunk, (i + 1) * chunk) for i in range(workers)]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_work, s, e) for s, e in ranges]
        results = [f.result() for f in futures]
    elapsed = time.perf_counter() - t0

    return {
        "benchmark": "cpu_multi_thread",
        "iterations": iterations,
        "workers": workers,
        "time_s": round(elapsed, 4),
        "ops_per_sec": int(iterations / elapsed) if elapsed > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Thread pool latency
# ═══════════════════════════════════════════════════════════════════════════

def run_thread_pool_latency(
    tasks: int = 1000,
    pool_size: int = 0,
) -> Dict[str, Any]:
    """Measure submit→complete latency for no-op tasks on the I/O pool."""
    from pyaccelerate.threads import get_pool, io_pool_size

    if pool_size <= 0:
        pool_size = io_pool_size()

    pool = get_pool()
    latencies: list[float] = []

    for _ in range(tasks):
        t0 = time.perf_counter()
        fut = pool.submit(lambda: None)
        fut.result()
        latencies.append(time.perf_counter() - t0)

    avg = sum(latencies) / len(latencies) if latencies else 0
    p95_idx = int(len(latencies) * 0.95)
    sorted_lat = sorted(latencies)

    return {
        "benchmark": "thread_pool_latency",
        "tasks": tasks,
        "pool_size": pool_size,
        "avg_latency_us": round(avg * 1_000_000, 1),
        "p95_latency_us": round(sorted_lat[p95_idx] * 1_000_000, 1) if sorted_lat else 0,
        "min_latency_us": round(sorted_lat[0] * 1_000_000, 1) if sorted_lat else 0,
        "max_latency_us": round(sorted_lat[-1] * 1_000_000, 1) if sorted_lat else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Memory bandwidth
# ═══════════════════════════════════════════════════════════════════════════

def run_memory_bandwidth(
    size_mb: int = 64,
    iterations: int = 10,
) -> Dict[str, Any]:
    """Measure sequential memory read/write bandwidth."""
    if iterations <= 0:
        return {
            "benchmark": "memory_bandwidth",
            "size_mb": size_mb,
            "iterations": 0,
            "write_gbps": 0,
            "read_gbps": 0,
            "error": "iterations must be > 0",
        }

    size = size_mb * 1024 * 1024
    buf = bytearray(size)

    # Write
    t0 = time.perf_counter()
    for _ in range(iterations):
        buf = bytearray(size)
    write_time = time.perf_counter() - t0

    # Read (hash to prevent optimization)
    t0 = time.perf_counter()
    for _ in range(iterations):
        _h = hashlib.md5(buf).digest()
    read_time = time.perf_counter() - t0

    total_bytes = size * iterations
    return {
        "benchmark": "memory_bandwidth",
        "size_mb": size_mb,
        "iterations": iterations,
        "write_gbps": round(total_bytes / write_time / (1024 ** 3), 2) if write_time > 0 else 0,
        "read_gbps": round(total_bytes / read_time / (1024 ** 3), 2) if read_time > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  GPU benchmark
# ═══════════════════════════════════════════════════════════════════════════

def run_gpu(
    size: int = 10_000_000,
    iterations: int = 100,
) -> Dict[str, Any]:
    """GPU compute throughput: element-wise operations on a large array.

    Falls back to a CPU result if no GPU backend is available.
    """
    from pyaccelerate.gpu import gpu_available, best_gpu

    if not gpu_available():
        return {
            "benchmark": "gpu_compute",
            "available": False,
            "note": "No usable GPU — skipped",
        }

    gpu = best_gpu()
    backend = gpu.backend if gpu else "none"

    # GPU memory topology (available regardless of compute backend)
    mem_info = {}
    if gpu:
        mem_info = {
            "dedicated_vram_gb": round(gpu.memory_gb, 2),
            "shared_vram_gb": round(gpu.shared_memory_gb, 2),
            "total_vram_gb": round(gpu.total_memory_gb, 2),
            "vulkan_version": gpu.vulkan_version or "",
            "is_discrete": gpu.is_discrete,
        }

    if backend == "cuda":
        result = _bench_cuda(size, iterations, gpu)
    elif backend == "opencl":
        result = _bench_opencl(size, iterations, gpu)
    elif backend == "intel":
        result = _bench_intel(size, iterations, gpu)
    else:
        result = {"benchmark": "gpu_compute", "available": False, "note": "Unsupported backend"}

    result.update(mem_info)
    return result


def _bench_cuda(size: int, iterations: int, gpu: Any) -> Dict[str, Any]:
    try:
        import cupy as cp  # type: ignore[import-untyped]
        with cp.cuda.Device(gpu._index):
            a = cp.random.random(size, dtype=cp.float32)
            b = cp.random.random(size, dtype=cp.float32)
            cp.cuda.Device(gpu._index).synchronize()

            t0 = time.perf_counter()
            for _ in range(iterations):
                c = a * b + a
            cp.cuda.Device(gpu._index).synchronize()
            elapsed = time.perf_counter() - t0

        ops = size * 2 * iterations  # mul + add
        return {
            "benchmark": "gpu_compute",
            "available": True,
            "backend": "cuda",
            "device": gpu.name,
            "elements": size,
            "iterations": iterations,
            "time_s": round(elapsed, 4),
            "gflops": round(ops / elapsed / 1e9, 2) if elapsed > 0 else 0,
        }
    except Exception as exc:
        return {"benchmark": "gpu_compute", "available": True, "error": str(exc)}


def _bench_opencl(size: int, iterations: int, gpu: Any) -> Dict[str, Any]:
    try:
        import numpy as np  # type: ignore[import-untyped]
        a = np.random.random(size).astype(np.float32)
        b = np.random.random(size).astype(np.float32)

        t0 = time.perf_counter()
        for _ in range(iterations):
            c = a * b + a
        elapsed = time.perf_counter() - t0

        ops = size * 2 * iterations
        return {
            "benchmark": "gpu_compute",
            "available": True,
            "backend": "opencl",
            "device": gpu.name,
            "elements": size,
            "iterations": iterations,
            "time_s": round(elapsed, 4),
            "gflops": round(ops / elapsed / 1e9, 2) if elapsed > 0 else 0,
            "note": "OpenCL benchmark uses numpy host-side as proxy",
        }
    except Exception as exc:
        return {"benchmark": "gpu_compute", "available": True, "error": str(exc)}


def _bench_intel(size: int, iterations: int, gpu: Any) -> Dict[str, Any]:
    try:
        import dpnp  # type: ignore[import-untyped]
        a = dpnp.random.random(size).astype(dpnp.float32)
        b = dpnp.random.random(size).astype(dpnp.float32)

        t0 = time.perf_counter()
        for _ in range(iterations):
            c = a * b + a
        elapsed = time.perf_counter() - t0

        ops = size * 2 * iterations
        return {
            "benchmark": "gpu_compute",
            "available": True,
            "backend": "intel",
            "device": gpu.name,
            "elements": size,
            "iterations": iterations,
            "time_s": round(elapsed, 4),
            "gflops": round(ops / elapsed / 1e9, 2) if elapsed > 0 else 0,
        }
    except Exception as exc:
        return {"benchmark": "gpu_compute", "available": True, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════
#  NPU benchmark
# ═══════════════════════════════════════════════════════════════════════════

def _create_bench_onnx_model(
    batch: int = 1,
    input_dim: int = 512,
    hidden_dim: int = 512,
    output_dim: int = 256,
    num_layers: int = 1,
) -> bytes:
    """Build a multi-layer MatMul→Relu ONNX model in memory (no file I/O)."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    rng = np.random.RandomState(42)
    initializers = []
    nodes = []
    prev_output = "X"
    prev_dim = input_dim

    for i in range(num_layers):
        is_last = (i == num_layers - 1)
        out_d = output_dim if is_last else hidden_dim
        w_name = f"W{i}"
        w = rng.randn(prev_dim, out_d).astype(np.float32) * 0.02
        tensor = onnx.TensorProto()
        tensor.name = w_name
        tensor.data_type = TensorProto.FLOAT
        tensor.dims.extend(w.shape)
        tensor.raw_data = w.tobytes()
        initializers.append(tensor)
        mm_out = f"mm_{i}"
        nodes.append(helper.make_node("MatMul", [prev_output, w_name], [mm_out]))
        if is_last:
            prev_output = mm_out
        else:
            relu_out = f"relu_{i}"
            nodes.append(helper.make_node("Relu", [mm_out], [relu_out]))
            prev_output = relu_out
        prev_dim = out_d

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [batch, input_dim])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch, output_dim])

    # Rename last output to Y
    nodes[-1].output[0] = "Y"

    graph = helper.make_graph(nodes, "bench_model", [X], [Y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()


def run_npu(
    batch: int = 1,
    input_dim: int = 512,
    iterations: int = 200,
    warmup: int = 20,
) -> Dict[str, Any]:
    """NPU inference benchmark: compare NPU vs CPU on a small ONNX model.

    Returns timing for both NPU and CPU backends, plus the speedup ratio.
    """
    import numpy as np
    from pyaccelerate.npu import npu_available

    if not npu_available():
        return {
            "benchmark": "npu_inference",
            "available": False,
            "note": "No usable NPU — skipped",
        }

    try:
        import onnx  # noqa: F401
    except ImportError:
        return {
            "benchmark": "npu_inference",
            "available": True,
            "note": "onnx package not installed — skipped (pip install onnx)",
        }

    model_bytes = _create_bench_onnx_model(batch=batch, input_dim=input_dim, num_layers=8)

    # Write to a temp file (ONNX Runtime / OpenVINO need a path)
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    tmp.write(model_bytes)
    tmp.close()
    model_path = tmp.name

    rng = np.random.RandomState(0)
    input_data = {"X": rng.randn(batch, input_dim).astype(np.float32)}

    results: Dict[str, Any] = {
        "benchmark": "npu_inference",
        "available": True,
        "batch": batch,
        "input_dim": input_dim,
        "model": f"MLP-8L ({input_dim}→512→256) × {iterations}",
        "iterations": iterations,
    }

    # ── NPU timing ──
    try:
        from pyaccelerate.npu.inference import InferenceSession
        sess_npu = InferenceSession(model_path, device="NPU")
        for _ in range(warmup):
            sess_npu.predict(input_data)
        t0 = time.perf_counter()
        for _ in range(iterations):
            sess_npu.predict(input_data)
        npu_elapsed = time.perf_counter() - t0
        results["npu_backend"] = sess_npu.backend_name
        results["npu_time_s"] = round(npu_elapsed, 4)
        results["npu_avg_ms"] = round(npu_elapsed / iterations * 1000, 3)
        results["npu_throughput"] = int(iterations / npu_elapsed) if npu_elapsed > 0 else 0
    except Exception as exc:
        results["npu_error"] = str(exc)
        npu_elapsed = 0.0

    # ── CPU timing (same model, CPU backend) ──
    try:
        from pyaccelerate.npu.inference import InferenceSession
        sess_cpu = InferenceSession(model_path, device="CPU", backend="cpu")
        for _ in range(warmup):
            sess_cpu.predict(input_data)
        t0 = time.perf_counter()
        for _ in range(iterations):
            sess_cpu.predict(input_data)
        cpu_elapsed = time.perf_counter() - t0
        results["cpu_time_s"] = round(cpu_elapsed, 4)
        results["cpu_avg_ms"] = round(cpu_elapsed / iterations * 1000, 3)
        results["cpu_throughput"] = int(iterations / cpu_elapsed) if cpu_elapsed > 0 else 0
    except Exception as exc:
        results["cpu_error"] = str(exc)
        cpu_elapsed = 0.0

    # ── Speedup ──
    if npu_elapsed > 0 and cpu_elapsed > 0:
        results["speedup"] = round(cpu_elapsed / npu_elapsed, 2)

    # Cleanup
    try:
        os.unlink(model_path)
    except OSError:
        pass

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Full suite
# ═══════════════════════════════════════════════════════════════════════════

def run_all(quick: bool = True) -> Dict[str, Any]:
    """Run all benchmarks and return a combined report.

    Parameters
    ----------
    quick : bool
        If True, use reduced iteration counts for a faster run (~5 s).
    """
    scale = 1 if quick else 5

    results: Dict[str, Any] = {}
    results["cpu_single"] = run_cpu(iterations=100_000 * scale)
    results["cpu_multi"] = run_cpu_multithread(iterations=100_000 * scale)
    results["thread_latency"] = run_thread_pool_latency(tasks=200 * scale)
    results["memory"] = run_memory_bandwidth(size_mb=16 * scale, iterations=3 * scale)
    results["gpu"] = run_gpu(size=1_000_000 * scale, iterations=20 * scale)
    results["npu"] = run_npu(iterations=50 * scale, warmup=10 * scale)

    return results
