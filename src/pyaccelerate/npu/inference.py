"""
pyaccelerate.npu.inference — High-level NPU inference API.

Provides a unified ``InferenceSession`` and ``run_inference()`` helper that
automatically selects the best available NPU backend (OpenVINO → ONNX Runtime
→ CPU fallback) and exposes a single, framework-agnostic interface.

Usage::

    from pyaccelerate.npu import run_inference, InferenceSession
    import numpy as np

    # One-shot convenience
    outputs = run_inference("model.onnx", {"input": np.zeros((1, 3, 224, 224))})

    # Reusable session
    sess = InferenceSession("model.onnx")
    print(sess.info())
    outputs = sess.predict({"input": data})
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("pyaccelerate.npu.inference")


class Backend(str, Enum):
    """NPU inference backends in preference order."""
    OPENVINO = "openvino"
    ONNXRT = "onnxrt"
    CPU = "cpu"


# ═══════════════════════════════════════════════════════════════════════════
#  Unified Inference Session
# ═══════════════════════════════════════════════════════════════════════════

class InferenceSession:
    """Framework-agnostic NPU inference session.

    Automatically picks the best backend:
      1. OpenVINO (if installed and NPU is present)
      2. ONNX Runtime with NPU EP
      3. ONNX Runtime CPU fallback

    Parameters
    ----------
    model_path : str
        Path to an ``.onnx``, ``.xml`` (OpenVINO IR), or ``.blob`` model.
    backend : Backend or str, optional
        Force a specific backend.  Auto-detected if *None*.
    device : str, optional
        Target device hint (e.g. ``"NPU"``, ``"CPU"``).
    config : dict, optional
        Extra backend-specific configuration.
    """

    def __init__(
        self,
        model_path: str,
        *,
        backend: Optional[Backend | str] = None,
        device: str = "NPU",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self._config = config or {}
        self._session: Any = None
        self._backend: Backend = Backend.CPU
        self._build_time_ms: float = 0.0

        if backend is not None:
            self._backend = Backend(backend)
            self._init_backend(self._backend)
        else:
            self._auto_select()

    # ── Backend initialisation ──────────────────────────────────────────

    def _auto_select(self) -> None:
        """Try backends in preference order."""
        for b in Backend:
            try:
                self._init_backend(b)
                self._backend = b
                log.info("Selected NPU backend: %s", b.value)
                return
            except Exception as exc:
                log.debug("Backend %s unavailable: %s", b.value, exc)
                continue
        raise RuntimeError(
            "No NPU inference backend available.  "
            "pip install openvino  or  pip install onnxruntime-directml"
        )

    def _init_backend(self, backend: Backend) -> None:
        t0 = time.perf_counter()
        if backend == Backend.OPENVINO:
            self._init_openvino()
        elif backend == Backend.ONNXRT:
            self._init_onnxrt()
        elif backend == Backend.CPU:
            self._init_cpu_fallback()
        self._build_time_ms = (time.perf_counter() - t0) * 1000.0

    def _init_openvino(self) -> None:
        from pyaccelerate.npu import openvino as ov_helpers
        if not ov_helpers.available() and self.device == "NPU":
            raise RuntimeError("OpenVINO NPU not available")
        model = ov_helpers.read_model(self.model_path)
        self._session = ov_helpers.compile_model(
            model, device=self.device, config=self._config or None
        )

    def _init_onnxrt(self) -> None:
        from pyaccelerate.npu import onnx_rt
        if not onnx_rt.onnxrt_available():
            raise ImportError("onnxruntime not installed")
        self._session = onnx_rt.create_session(
            self.model_path,
            fallback_cpu=(self.device == "NPU"),
            **(self._config or {}),
        )

    def _init_cpu_fallback(self) -> None:
        from pyaccelerate.npu import onnx_rt
        if not onnx_rt.onnxrt_available():
            raise ImportError("onnxruntime not installed")
        self._session = onnx_rt.create_session(
            self.model_path,
            ep="CPUExecutionProvider",
            fallback_cpu=True,
        )

    # ── Inference ────────────────────────────────────────────────────────

    def predict(
        self,
        inputs: Dict[str, Any],
        *,
        output_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Run inference and return output tensors as a dict.

        Parameters
        ----------
        inputs
            Dict mapping input tensor names to numpy arrays.
        output_names
            Specific outputs to return.  None = all.

        Returns
        -------
        dict
            Output name → numpy array.
        """
        if self._backend == Backend.OPENVINO:
            from pyaccelerate.npu import openvino as ov_helpers
            return ov_helpers.infer(self._session, inputs)

        # ONNX Runtime path
        from pyaccelerate.npu import onnx_rt
        raw = onnx_rt.run_inference_onnx(
            self._session, inputs,
            output_names=list(output_names) if output_names else None,
        )
        # Map to dict using session output metadata
        out_info = self._session.get_outputs()
        result: Dict[str, Any] = {}
        for i, arr in enumerate(raw):
            name = out_info[i].name if i < len(out_info) else f"output_{i}"
            result[name] = arr
        return result

    def predict_batch(
        self,
        inputs_batch: Sequence[Dict[str, Any]],
        *,
        max_requests: int = 0,
    ) -> List[Dict[str, Any]]:
        """Run batched / async inference (OpenVINO pipelining or sequential).

        Parameters
        ----------
        inputs_batch
            Sequence of input dicts.
        max_requests
            Max concurrent requests (OpenVINO async). 0 = optimal.

        Returns
        -------
        list[dict]
            One output dict per input.
        """
        if self._backend == Backend.OPENVINO:
            from pyaccelerate.npu import openvino as ov_helpers
            return ov_helpers.infer_async(
                self._session, inputs_batch, max_requests=max_requests,
            )

        # Sequential fallback for ONNX Runtime
        return [self.predict(inp) for inp in inputs_batch]

    # ── Info / metadata ──────────────────────────────────────────────────

    @property
    def backend_name(self) -> str:
        return self._backend.value

    def info(self) -> Dict[str, Any]:
        """Session metadata."""
        d: Dict[str, Any] = {
            "model_path": self.model_path,
            "backend": self._backend.value,
            "device": self.device,
            "build_time_ms": round(self._build_time_ms, 2),
        }
        if self._backend == Backend.OPENVINO:
            from pyaccelerate.npu import openvino as ov_helpers
            d["model_info"] = ov_helpers.model_info(self._session)
        elif self._session is not None:
            from pyaccelerate.npu import onnx_rt
            d["session_info"] = onnx_rt.get_session_info(self._session)
        return d


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience one-shot function
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(
    model_path: str,
    inputs: Dict[str, Any],
    *,
    backend: Optional[str] = None,
    device: str = "NPU",
    output_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """One-shot NPU inference.

    Creates a session, runs one prediction, returns results.
    For repeated use, prefer ``InferenceSession``.

    Parameters
    ----------
    model_path : str
        Path to an ``.onnx`` or ``.xml`` model.
    inputs : dict
        Input name → numpy array.
    backend : str, optional
        ``"openvino"``, ``"onnxrt"``, or ``"cpu"``.  Auto if None.
    device : str
        Device hint.  ``"NPU"`` (default) or ``"CPU"``.
    output_names : list[str], optional
        Subset of model outputs.

    Returns
    -------
    dict
        Output name → numpy array.
    """
    sess = InferenceSession(
        model_path, backend=backend, device=device,
    )
    return sess.predict(inputs, output_names=output_names)
