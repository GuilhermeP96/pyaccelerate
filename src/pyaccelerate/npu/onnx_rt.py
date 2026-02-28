"""
pyaccelerate.npu.onnx_rt — ONNX Runtime bridge for NPU inference.

Provides a thin abstraction over ``onnxruntime.InferenceSession`` that
automatically selects the best NPU Execution Provider.

EP priority (first available wins):
  1. OpenVINOExecutionProvider  — Intel NPU
  2. QNNExecutionProvider       — Qualcomm Hexagon NPU
  3. DmlExecutionProvider       — Any Windows NPU via DirectML
  4. CoreMLExecutionProvider    — Apple Neural Engine
  5. CPUExecutionProvider       — fallback (no NPU)

Usage::

    from pyaccelerate.npu.onnx_rt import create_session, list_npu_eps

    eps = list_npu_eps()          # ["DmlExecutionProvider", ...]
    sess = create_session("model.onnx")
    output = sess.run(None, {"input": data})
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("pyaccelerate.npu.onnx_rt")


# ═══════════════════════════════════════════════════════════════════════════
#  EP Discovery
# ═══════════════════════════════════════════════════════════════════════════

# Ordered by NPU preference
_NPU_EPS: List[str] = [
    "OpenVINOExecutionProvider",
    "QNNExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
]


def _get_ort():
    """Lazy import of onnxruntime."""
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
        return ort
    except ImportError:
        return None


def list_all_eps() -> List[str]:
    """Return all ONNX Runtime EPs installed on this system."""
    ort = _get_ort()
    if ort is None:
        return []
    return ort.get_available_providers()


def list_npu_eps() -> List[str]:
    """Return only the NPU-capable EPs available."""
    all_eps = list_all_eps()
    return [ep for ep in _NPU_EPS if ep in all_eps]


def best_ep() -> Optional[str]:
    """Return the single best NPU EP, or None."""
    eps = list_npu_eps()
    return eps[0] if eps else None


def onnxrt_available() -> bool:
    """True if onnxruntime is installed."""
    return _get_ort() is not None


def npu_ep_available() -> bool:
    """True if at least one NPU EP is available."""
    return bool(list_npu_eps())


# ═══════════════════════════════════════════════════════════════════════════
#  Session Factory
# ═══════════════════════════════════════════════════════════════════════════

def _build_session_options(
    *,
    graph_optimization_level: int = 99,
    inter_op_threads: int = 0,
    intra_op_threads: int = 0,
    enable_profiling: bool = False,
    log_severity: int = 3,
) -> Any:
    """Build ``onnxruntime.SessionOptions`` with sensible NPU defaults."""
    ort = _get_ort()
    if ort is None:
        raise ImportError("onnxruntime is not installed")

    opts = ort.SessionOptions()

    # Graph optimization (0=disabled, 1=basic, 2=extended, 99=all)
    level_map = {
        0: ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        1: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        2: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        99: ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    opts.graph_optimization_level = level_map.get(
        graph_optimization_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    if inter_op_threads:
        opts.inter_op_num_threads = inter_op_threads
    if intra_op_threads:
        opts.intra_op_num_threads = intra_op_threads

    opts.enable_profiling = enable_profiling
    opts.log_severity_level = log_severity

    return opts


def _build_provider_options(ep_name: str) -> Dict[str, Any]:
    """Per-EP options for optimal NPU throughput."""
    opts: Dict[str, Any] = {}

    if ep_name == "OpenVINOExecutionProvider":
        opts["device_type"] = "NPU"
        cache = os.environ.get("PYACC_OPENVINO_CACHE", "")
        if cache:
            opts["cache_dir"] = cache

    elif ep_name == "DmlExecutionProvider":
        # DirectML default device (0 = first DML device)
        opts["device_id"] = int(os.environ.get("PYACC_DML_DEVICE", "0"))

    elif ep_name == "QNNExecutionProvider":
        qnn_lib = os.environ.get("PYACC_QNN_LIB_PATH", "")
        if qnn_lib:
            opts["backend_path"] = qnn_lib

    return opts


def create_session(
    model_path: str,
    *,
    ep: Optional[str] = None,
    fallback_cpu: bool = True,
    session_options: Optional[Any] = None,
    provider_options: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Create an ``onnxruntime.InferenceSession`` targeting the best NPU EP.

    Parameters
    ----------
    model_path : str
        Path to an ``.onnx`` model file.
    ep : str, optional
        Force a specific Execution Provider.  Auto-selected if *None*.
    fallback_cpu : bool
        If True (default), append ``CPUExecutionProvider`` as fallback.
    session_options : optional
        Pre-built ``SessionOptions``.  If None, built with NPU defaults.
    provider_options : dict, optional
        Extra provider options merged on top of defaults.
    **kwargs
        Forwarded to ``_build_session_options()`` when *session_options* is
        None.

    Returns
    -------
    onnxruntime.InferenceSession
    """
    ort = _get_ort()
    if ort is None:
        raise ImportError(
            "onnxruntime is not installed.  pip install onnxruntime-directml "
            "or pip install onnxruntime-openvino"
        )

    # EP selection
    chosen_ep = ep or best_ep()
    eps_list: List[str] = []
    ep_opts_list: List[Dict[str, Any]] = []

    if chosen_ep:
        merged_opts = _build_provider_options(chosen_ep)
        if provider_options:
            merged_opts.update(provider_options)
        eps_list.append(chosen_ep)
        ep_opts_list.append(merged_opts)

    if fallback_cpu and "CPUExecutionProvider" not in eps_list:
        eps_list.append("CPUExecutionProvider")
        ep_opts_list.append({})

    # Session options
    if session_options is None:
        session_options = _build_session_options(**kwargs)

    log.info("Creating ONNX RT session: model=%s, EPs=%s", model_path, eps_list)

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=eps_list,
        provider_options=ep_opts_list if ep_opts_list else None,
    )

    actual_ep = session.get_providers()[0] if session.get_providers() else "none"
    log.info("Session using EP: %s", actual_ep)

    return session


def run_inference_onnx(
    session: Any,
    inputs: Dict[str, Any],
    output_names: Optional[Sequence[str]] = None,
) -> List[Any]:
    """Run inference on an ONNX Runtime session.

    Parameters
    ----------
    session
        An ``onnxruntime.InferenceSession``.
    inputs
        Dict mapping input names to numpy arrays.
    output_names
        Specific outputs to fetch.  None = all.

    Returns
    -------
    list
        Model outputs as numpy arrays.
    """
    return session.run(output_names, inputs)


def get_session_info(session: Any) -> Dict[str, Any]:
    """Retrieve metadata from an active ONNX Runtime session."""
    info: Dict[str, Any] = {}
    try:
        info["providers"] = session.get_providers()
        info["active_provider"] = session.get_providers()[0] if session.get_providers() else "none"
        info["inputs"] = [
            {"name": inp.name, "shape": inp.shape, "type": inp.type}
            for inp in session.get_inputs()
        ]
        info["outputs"] = [
            {"name": out.name, "shape": out.shape, "type": out.type}
            for out in session.get_outputs()
        ]
    except Exception as exc:
        info["error"] = str(exc)
    return info
