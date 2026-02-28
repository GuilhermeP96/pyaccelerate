"""
pyaccelerate.npu — Neural Processing Unit (NPU) detection, dispatch & inference.

Supports modern NPU hardware across all major vendors:

+-------------------+-----------------------------------+---------------------------+
| Vendor            | Hardware                          | Framework                 |
+-------------------+-----------------------------------+---------------------------+
| Intel             | Meteor Lake, Arrow Lake, Lunar    | OpenVINO, intel-npu-accel |
| AMD               | Ryzen AI (XDNA / Phoenix / Hawk) | ONNX Runtime (Vitis AI)   |
| Qualcomm          | Snapdragon X (Hexagon DSP)        | ONNX Runtime (QNN EP)     |
| Apple             | Neural Engine (M1-M4)             | Core ML (coremltools)     |
| Windows (generic) | Any NPU via DirectML              | ONNX Runtime (DML EP)     |
+-------------------+-----------------------------------+---------------------------+

The common bridge is **ONNX Runtime** with vendor-specific Execution Providers
(EPs).  For Intel, **OpenVINO** also works as a native path.

Quick start::

    from pyaccelerate.npu import detect_all, npu_available, run_inference

    npus = detect_all()
    if npu_available():
        result = run_inference("model.onnx", input_data)
"""

from pyaccelerate.npu.detector import (
    NPUDevice,
    detect_all,
    best_npu,
    npu_available,
    get_npu_info,
    get_all_npus_info,
    get_install_hint,
    reset_cache,
)
from pyaccelerate.npu.inference import run_inference, InferenceSession

__all__ = [
    "NPUDevice",
    "detect_all",
    "best_npu",
    "npu_available",
    "get_npu_info",
    "get_all_npus_info",
    "get_install_hint",
    "reset_cache",
    "run_inference",
    "InferenceSession",
]
