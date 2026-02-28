"""
pyaccelerate.npu.openvino — Intel NPU helpers via OpenVINO Toolkit.

Wraps the OpenVINO 2024+ API for compiling and running models on the Intel
Neural Processing Unit (NPU).  Works on Meteor Lake, Arrow Lake, Lunar Lake,
and newer Intel platforms.

Usage::

    from pyaccelerate.npu.openvino import compile_model, infer, available

    if available():
        compiled = compile_model("model.xml")
        results  = infer(compiled, {"input": np_array})
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("pyaccelerate.npu.openvino")


# ═══════════════════════════════════════════════════════════════════════════
#  Lazy import
# ═══════════════════════════════════════════════════════════════════════════

def _get_ov():
    """Lazy import of openvino."""
    try:
        import openvino as ov  # type: ignore[import-untyped]
        return ov
    except ImportError:
        return None


def _get_core():
    """Get/create a singleton ``openvino.Core()``."""
    ov = _get_ov()
    if ov is None:
        return None
    if not hasattr(_get_core, "_instance"):
        _get_core._instance = ov.Core()
    return _get_core._instance


# ═══════════════════════════════════════════════════════════════════════════
#  Availability
# ═══════════════════════════════════════════════════════════════════════════

def available() -> bool:
    """True if OpenVINO is installed *and* an NPU device is present."""
    core = _get_core()
    if core is None:
        return False
    try:
        return "NPU" in core.available_devices
    except Exception:
        return False


def list_devices() -> List[str]:
    """List all OpenVINO devices (CPU, GPU, NPU, etc.)."""
    core = _get_core()
    if core is None:
        return []
    return core.available_devices


def get_npu_properties() -> Dict[str, str]:
    """Retrieve NPU device properties via OpenVINO."""
    core = _get_core()
    if core is None or not available():
        return {}

    props: Dict[str, str] = {}
    prop_keys = [
        "FULL_DEVICE_NAME",
        "NPU_DRIVER_VERSION",
        "DEVICE_ARCHITECTURE",
        "OPTIMAL_NUMBER_OF_INFER_REQUESTS",
        "RANGE_FOR_ASYNC_INFER_REQUESTS",
        "SUPPORTED_PROPERTIES",
    ]
    for key in prop_keys:
        try:
            props[key] = str(core.get_property("NPU", key))
        except Exception:
            pass
    return props


# ═══════════════════════════════════════════════════════════════════════════
#  Model compilation
# ═══════════════════════════════════════════════════════════════════════════

def read_model(
    model_path: str,
    weights_path: Optional[str] = None,
) -> Any:
    """Read an IR (.xml) or ONNX model into an ``openvino.Model``.

    Parameters
    ----------
    model_path
        Path to .xml (OpenVINO IR) or .onnx model.
    weights_path
        Path to .bin weights.  Auto-detected for IR models.
    """
    core = _get_core()
    if core is None:
        raise ImportError("openvino is not installed.  pip install openvino")

    if weights_path:
        return core.read_model(model=model_path, weights=weights_path)
    return core.read_model(model=model_path)


def compile_model(
    model: Any,
    *,
    device: str = "NPU",
    config: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
) -> Any:
    """Compile a model for execution on the NPU.

    Parameters
    ----------
    model
        An ``openvino.Model`` **or** a path string (.xml / .onnx).
    device
        Target device.  ``"NPU"`` by default.
    config
        Extra compilation config (e.g. ``{"NPU_COMPILATION_MODE": "..."}``).
    cache_dir
        Directory for compiled-model cache.  Speeds up subsequent loads.

    Returns
    -------
    openvino.CompiledModel
    """
    core = _get_core()
    if core is None:
        raise ImportError("openvino is not installed")

    # Accept path → read model first
    if isinstance(model, str):
        model = read_model(model)

    conf: Dict[str, Any] = {}
    if cache_dir:
        conf["CACHE_DIR"] = cache_dir
    elif os.environ.get("PYACC_OPENVINO_CACHE"):
        conf["CACHE_DIR"] = os.environ["PYACC_OPENVINO_CACHE"]

    if config:
        conf.update(config)

    log.info("Compiling model for device=%s, config=%s", device, conf or "{}")
    compiled = core.compile_model(model, device_name=device, config=conf or None)
    log.info("Compiled successfully — optimal infer requests: %s",
             compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
    return compiled


# ═══════════════════════════════════════════════════════════════════════════
#  Inference
# ═══════════════════════════════════════════════════════════════════════════

def infer(
    compiled_model: Any,
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """Run synchronous inference on a compiled model.

    Parameters
    ----------
    compiled_model
        ``openvino.CompiledModel`` (from ``compile_model``).
    inputs
        Dict mapping input tensor names to numpy arrays.

    Returns
    -------
    dict
        Output tensors keyed by output name.
    """
    request = compiled_model.create_infer_request()
    request.infer(inputs)

    outputs: Dict[str, Any] = {}
    for output in compiled_model.outputs:
        name = output.any_name
        outputs[name] = request.get_output_tensor(
            list(compiled_model.outputs).index(output)
        ).data.copy()

    return outputs


def infer_async(
    compiled_model: Any,
    inputs_batch: Sequence[Dict[str, Any]],
    *,
    max_requests: int = 0,
) -> List[Dict[str, Any]]:
    """Run asynchronous inference with request pipelining.

    Parameters
    ----------
    compiled_model
        ``openvino.CompiledModel``.
    inputs_batch
        Sequence of input dicts (one per inference).
    max_requests
        Max concurrent async requests.  0 = optimal (device default).

    Returns
    -------
    list[dict]
        One output dict per input.
    """
    ov = _get_ov()
    if ov is None:
        raise ImportError("openvino is not installed")

    if max_requests <= 0:
        try:
            max_requests = compiled_model.get_property(
                "OPTIMAL_NUMBER_OF_INFER_REQUESTS"
            )
        except Exception:
            max_requests = 4

    # Create an AsyncInferQueue
    infer_queue = ov.AsyncInferQueue(compiled_model, max_requests)

    results: List[Optional[Dict[str, Any]]] = [None] * len(inputs_batch)

    def _callback(request, userdata):
        idx = userdata
        out_dict: Dict[str, Any] = {}
        for i, output in enumerate(compiled_model.outputs):
            name = output.any_name
            out_dict[name] = request.get_output_tensor(i).data.copy()
        results[idx] = out_dict

    infer_queue.set_callback(_callback)

    for idx, inp in enumerate(inputs_batch):
        infer_queue.start_async(inp, userdata=idx)

    infer_queue.wait_all()
    return results  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════════════════

def model_info(model: Any) -> Dict[str, Any]:
    """Get input/output metadata from an OpenVINO model.

    Parameters
    ----------
    model
        ``openvino.Model`` or ``openvino.CompiledModel``.

    Returns
    -------
    dict
        Keys: ``inputs``, ``outputs``.
    """
    info: Dict[str, Any] = {}
    try:
        info["inputs"] = [
            {
                "name": inp.any_name,
                "shape": list(inp.shape) if hasattr(inp, "shape") else "dynamic",
                "type": str(inp.element_type),
            }
            for inp in model.inputs
        ]
        info["outputs"] = [
            {
                "name": out.any_name,
                "shape": list(out.shape) if hasattr(out, "shape") else "dynamic",
                "type": str(out.element_type),
            }
            for out in model.outputs
        ]
    except Exception as exc:
        info["error"] = str(exc)
    return info


def benchmark_npu(
    model_path: str,
    *,
    iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """Quick latency benchmark of a model on the NPU.

    Returns
    -------
    dict
        Keys: ``mean_ms``, ``min_ms``, ``max_ms``, ``p95_ms``, ``throughput_fps``.
    """
    import time
    import numpy as np

    model = read_model(model_path)
    compiled = compile_model(model, device="NPU")

    # Build dummy inputs
    inputs: Dict[str, Any] = {}
    for inp in model.inputs:
        shape = list(inp.shape)
        # Replace dynamic dims with 1
        shape = [s if isinstance(s, int) and s > 0 else 1 for s in shape]
        dtype_map = {"f32": np.float32, "f16": np.float16, "i32": np.int32, "i64": np.int64}
        etype = str(inp.element_type)
        np_dtype = dtype_map.get(etype, np.float32)
        inputs[inp.any_name] = np.random.randn(*shape).astype(np_dtype)

    request = compiled.create_infer_request()

    # Warmup
    for _ in range(warmup):
        request.infer(inputs)

    # Timed runs
    latencies: List[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        request.infer(inputs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    p95_idx = int(len(latencies) * 0.95)

    return {
        "mean_ms": round(mean_ms, 3),
        "min_ms": round(latencies[0], 3),
        "max_ms": round(latencies[-1], 3),
        "p95_ms": round(latencies[p95_idx], 3),
        "throughput_fps": round(1000.0 / mean_ms, 1) if mean_ms > 0 else 0.0,
    }
