"""
pyaccelerate.gpu — Multi-vendor GPU detection, ranking and dispatch.

Sub-modules
-----------
- ``detector``   : Enumerate GPUs across all backends (CUDA, OpenCL, Intel oneAPI, OS-level)
- ``cuda``       : CUDA/CuPy-specific helpers
- ``opencl``     : PyOpenCL helpers
- ``intel``      : Intel oneAPI / dpctl helpers
- ``dispatch``   : Multi-GPU workload dispatcher and load balancer

Quick start::

    from pyaccelerate.gpu import detect_all, best_gpu, gpu_available, dispatch

    gpus = detect_all()
    if gpu_available():
        result = dispatch(my_kernel, data, strategy="round-robin")
"""

from pyaccelerate.gpu.detector import (
    GPUDevice,
    detect_all,
    best_gpu,
    gpu_available,
    get_gpu_info,
    get_all_gpus_info,
    get_install_hint,
    reset_cache,
)
from pyaccelerate.gpu.dispatch import dispatch, multi_gpu_map
from pyaccelerate.gpu.hasher import gpu_hash_file, gpu_hash_batch, gpu_hash_available

__all__ = [
    "GPUDevice",
    "detect_all",
    "best_gpu",
    "gpu_available",
    "get_gpu_info",
    "get_all_gpus_info",
    "get_install_hint",
    "reset_cache",
    "dispatch",
    "multi_gpu_map",
    "gpu_hash_file",
    "gpu_hash_batch",
    "gpu_hash_available",
]
