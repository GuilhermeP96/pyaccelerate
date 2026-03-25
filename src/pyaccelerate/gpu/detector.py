"""
pyaccelerate.gpu.detector — Multi-vendor, multi-backend GPU enumeration.

Enumerates ALL GPUs on the system across:
  - CuPy / CUDA  (NVIDIA)
  - PyOpenCL      (NVIDIA, AMD, Intel)
  - Intel oneAPI / dpctl (Intel, including Arc)
  - OS-level fallback (display-only, no compute)

GPUs are ranked by a composite *score* (VRAM + compute units + discrete bonus).
Detection runs once and is cached; call ``reset_cache()`` to force re-scan.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("pyaccelerate.gpu.detector")


# ═══════════════════════════════════════════════════════════════════════════
#  GPU Device Descriptor
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GPUDevice:
    """Represents one detected GPU compute device."""

    name: str = ""
    backend: str = ""          # "cuda", "opencl", "intel", "vulkan", "none"
    vendor: str = ""           # "NVIDIA", "Intel", "AMD", "unknown"
    memory_bytes: int = 0      # VRAM (dedicated/global memory) in bytes
    shared_memory_bytes: int = 0  # Shared VRAM (system RAM mapped for GPU) in bytes
    compute_units: int = 0     # SMs / CUs / EUs
    is_discrete: bool = False  # discrete vs integrated
    vulkan_version: str = ""   # e.g. "1.2.154" if Vulkan-capable
    _module: Any = None        # runtime handle (cupy, pyopencl ctx, dpctl device)
    _index: int = 0            # device ordinal in its backend

    # ── Extended attributes (v0.9) ──────────────────────────────────────
    architecture: str = ""         # "Turing", "Ampere", "Ada Lovelace", "RDNA 3" …
    cuda_capability: str = ""      # "7.5", "8.6" (NVIDIA CUDA compute capability)
    cuda_cores: int = 0            # Total CUDA cores (NVIDIA) or Stream Processors (AMD)
    tensor_cores: int = 0          # Total Tensor cores (RTX / Volta / data-center)
    rt_cores: int = 0              # Total RT cores (hardware ray-tracing units)
    has_tensor: bool = False       # Tensor / matrix acceleration present
    has_raytracing: bool = False   # Hardware ray-tracing present
    has_nvenc: bool = False        # Hardware video encoder (NVIDIA NVENC / AMD VCN)
    has_nvdec: bool = False        # Hardware video decoder (NVIDIA NVDEC / AMD VCN)
    clock_mhz: int = 0            # Base / current clock (MHz)
    boost_clock_mhz: int = 0      # Boost clock (MHz)
    memory_clock_mhz: int = 0     # Memory clock (MHz)
    memory_type: str = ""          # "GDDR6", "GDDR6X", "HBM2e", "HBM3" …
    memory_bus_width: int = 0      # Memory bus width (bits)
    memory_bandwidth_gbps: float = 0.0  # Effective memory bandwidth (GB/s)
    pcie_gen: int = 0              # PCIe generation (3, 4, 5)
    pcie_width: int = 0            # PCIe link width (16, 8 …)
    driver_version: str = ""       # GPU driver version string
    cuda_driver_version: str = ""  # CUDA toolkit / driver API version
    power_limit_w: int = 0         # TDP / power limit (watts)
    l2_cache_bytes: int = 0        # L2 cache size (bytes)
    copy_engines: int = 0          # Async DMA copy engines

    @property
    def memory_gb(self) -> float:
        return self.memory_bytes / (1024 ** 3) if self.memory_bytes else 0.0

    @property
    def shared_memory_gb(self) -> float:
        return self.shared_memory_bytes / (1024 ** 3) if self.shared_memory_bytes else 0.0

    @property
    def total_memory_bytes(self) -> int:
        """Total addressable memory: dedicated + shared."""
        return self.memory_bytes + self.shared_memory_bytes

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_bytes / (1024 ** 3) if self.total_memory_bytes else 0.0

    @property
    def score(self) -> int:
        """Composite power score for ranking.

        Discrete GPUs get a large bonus.  Integrated GPUs with shared VRAM
        get a smaller bonus proportional to usable shared memory (capped at
        half the weight of dedicated VRAM to avoid over-ranking iGPUs).
        Modern features (tensor, RT) add moderate bonuses.
        """
        s = self.memory_bytes // (1024 * 1024)  # MB of dedicated VRAM
        # Shared VRAM counts at half weight (system RAM is slower than GDDR)
        if self.shared_memory_bytes and not self.is_discrete:
            s += self.shared_memory_bytes // (1024 * 1024) // 2
        s += self.compute_units * 50
        if self.is_discrete:
            s += 100_000
        if self.has_tensor:
            s += 50_000
        if self.has_raytracing:
            s += 10_000
        return s

    @property
    def usable(self) -> bool:
        """True if a compute backend is available (not just OS-level name)."""
        return self.backend != "none"

    @property
    def features(self) -> List[str]:
        """List of hardware capability flags for this GPU."""
        f: List[str] = []
        if self.usable:
            f.append("compute")
        if self.has_tensor:
            f.append("tensor")
        if self.has_raytracing:
            f.append("raytracing")
        if self.has_nvenc:
            f.append("hw_encode")
        if self.has_nvdec:
            f.append("hw_decode")
        if self.cuda_capability:
            f.append(f"cuda_{self.cuda_capability}")
        if self.architecture:
            f.append(self.architecture.lower().replace(" ", "_"))
        if self.copy_engines:
            f.append(f"copy_engines×{self.copy_engines}")
        return f

    def short_label(self) -> str:
        mem = f"{self.memory_gb:.1f} GB" if self.memory_bytes else "?"
        extra = ""
        if self.shared_memory_bytes:
            extra = f" +{self.shared_memory_gb:.1f} GB shared"
        arch = f" {self.architecture}" if self.architecture else ""
        vk = " Vulkan" if self.vulkan_version else ""
        return f"{self.name} ({self.backend.upper()}, {mem}{extra}{arch}{vk})"

    def as_dict(self) -> Dict[str, str]:
        d: Dict[str, str] = {
            "name": self.name,
            "backend": self.backend,
            "vendor": self.vendor,
            "memory": f"{self.memory_gb:.1f} GB",
            "compute_units": str(self.compute_units),
            "discrete": str(self.is_discrete),
            "score": str(self.score),
            "usable": str(self.usable),
        }
        if self.shared_memory_bytes:
            d["shared_memory"] = f"{self.shared_memory_gb:.1f} GB"
            d["total_memory"] = f"{self.total_memory_gb:.1f} GB"
        if self.vulkan_version:
            d["vulkan_version"] = self.vulkan_version
        if self.architecture:
            d["architecture"] = self.architecture
        if self.cuda_capability:
            d["cuda_capability"] = self.cuda_capability
        if self.cuda_cores:
            d["cuda_cores"] = str(self.cuda_cores)
        if self.tensor_cores:
            d["tensor_cores"] = str(self.tensor_cores)
        if self.rt_cores:
            d["rt_cores"] = str(self.rt_cores)
        if self.has_tensor:
            d["has_tensor"] = "True"
        if self.has_raytracing:
            d["has_raytracing"] = "True"
        if self.has_nvenc:
            d["has_hw_encode"] = "True"
        if self.has_nvdec:
            d["has_hw_decode"] = "True"
        if self.clock_mhz:
            d["clock_mhz"] = str(self.clock_mhz)
        if self.boost_clock_mhz:
            d["boost_clock_mhz"] = str(self.boost_clock_mhz)
        if self.memory_type:
            d["memory_type"] = self.memory_type
        if self.memory_bus_width:
            d["memory_bus_width"] = f"{self.memory_bus_width}-bit"
        if self.memory_bandwidth_gbps:
            d["memory_bandwidth"] = f"{self.memory_bandwidth_gbps:.0f} GB/s"
        if self.pcie_gen:
            d["pcie"] = f"Gen{self.pcie_gen} x{self.pcie_width}" if self.pcie_width else f"Gen{self.pcie_gen}"
        if self.driver_version:
            d["driver_version"] = self.driver_version
        if self.power_limit_w:
            d["power_limit"] = f"{self.power_limit_w} W"
        if self.copy_engines:
            d["copy_engines"] = str(self.copy_engines)
        d["features"] = ", ".join(self.features) if self.features else "none"
        return d


# ═══════════════════════════════════════════════════════════════════════════
#  Cache & lock
# ═══════════════════════════════════════════════════════════════════════════

_all_gpus: List[GPUDevice] = []
_best_gpu: Optional[GPUDevice] = None
_detected = False
_detect_lock = threading.Lock()


def reset_cache() -> None:
    """Force re-detection on next call to ``detect_all()``."""
    global _all_gpus, _best_gpu, _detected
    with _detect_lock:
        _all_gpus = []
        _best_gpu = None
        _detected = False


# ═══════════════════════════════════════════════════════════════════════════
#  Vendor heuristic
# ═══════════════════════════════════════════════════════════════════════════

def _vendor_from_name(name: str) -> Tuple[str, bool]:
    """Guess vendor and discrete flag from device name string."""
    nl = name.lower()
    if any(k in nl for k in ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla", "a100", "h100")):
        return "NVIDIA", True
    if any(k in nl for k in ("radeon", "amd", "rx ", "vega", "instinct")):
        return "AMD", True
    if any(k in nl for k in ("intel", "uhd", "iris", "arc")):
        is_discrete = "arc" in nl
        return "Intel", is_discrete
    if any(k in nl for k in ("apple", "m1", "m2", "m3", "m4")):
        return "Apple", True
    # ARM mobile GPUs
    if any(k in nl for k in ("adreno", "qualcomm")):
        return "Qualcomm", False
    if any(k in nl for k in ("mali", "immortalis")):
        return "ARM", False
    if any(k in nl for k in ("xclipse", "samsung gpu")):
        return "Samsung", False
    if any(k in nl for k in ("powervr", "imagination")):
        return "Imagination", False
    if any(k in nl for k in ("maleoon",)):
        return "HiSilicon", False
    # SBC / IoT GPUs
    if any(k in nl for k in ("videocore", "vc4", "v3d")):
        return "Broadcom", False
    if any(k in nl for k in ("tegra", "jetson")):
        return "NVIDIA", False
    if any(k in nl for k in ("vivante", "galcore", "gc7000", "gc nano")):
        return "Vivante", False
    return "unknown", False


# ═══════════════════════════════════════════════════════════════════════════
#  Architecture databases — NVIDIA & AMD
# ═══════════════════════════════════════════════════════════════════════════

# (major, minor) → (arch_name, cuda_cores_per_sm, tensor_per_sm, rt_per_sm)
# tensor_per_sm / rt_per_sm = 0 means the *arch* doesn't have them at all.
# Non-zero means the arch supports them but the specific SKU may not (e.g. GTX 16xx).
_NVIDIA_ARCH: Dict[Tuple[int, int], Tuple[str, int, int, int]] = {
    (3, 0): ("Kepler",        192, 0, 0),
    (3, 2): ("Kepler",        192, 0, 0),
    (3, 5): ("Kepler",        192, 0, 0),
    (3, 7): ("Kepler",        192, 0, 0),
    (5, 0): ("Maxwell",       128, 0, 0),
    (5, 2): ("Maxwell",       128, 0, 0),
    (5, 3): ("Maxwell",       128, 0, 0),
    (6, 0): ("Pascal",         64, 0, 0),
    (6, 1): ("Pascal",        128, 0, 0),
    (6, 2): ("Pascal",        128, 0, 0),
    (7, 0): ("Volta",          64, 8, 0),   # V100: always has tensor
    (7, 2): ("Volta",          64, 8, 0),   # Jetson Xavier
    (7, 5): ("Turing",         64, 8, 1),   # RTX 20xx: tensor+RT; GTX 16xx: neither
    (8, 0): ("Ampere",         64, 4, 0),   # A100 (data-center, no RT)
    (8, 6): ("Ampere",        128, 4, 1),   # RTX 30xx
    (8, 7): ("Ampere",        128, 4, 1),   # Jetson Orin
    (8, 9): ("Ada Lovelace",  128, 4, 1),   # RTX 40xx
    (9, 0): ("Hopper",        128, 4, 0),   # H100 (data-center, no RT)
    (10, 0): ("Blackwell",    128, 4, 1),   # RTX 50xx / B100
    (10, 2): ("Blackwell",    128, 4, 1),   # B200
}

# AMD GPU name patterns → (architecture, has_ray_accel, has_vcn_encode)
_AMD_ARCH_PATTERNS: List[Tuple[str, str, bool, bool]] = [
    ("rx 9",       "RDNA 4",  True,  True),
    ("rx 79",      "RDNA 3",  True,  True),
    ("rx 78",      "RDNA 3",  True,  True),
    ("rx 76",      "RDNA 3",  True,  True),
    ("rx 75",      "RDNA 3",  True,  True),
    ("rx 69",      "RDNA 2",  True,  True),
    ("rx 68",      "RDNA 2",  True,  True),
    ("rx 67",      "RDNA 2",  True,  True),
    ("rx 66",      "RDNA 2",  True,  True),
    ("rx 65",      "RDNA 2",  True,  True),
    ("rx 64",      "RDNA 2",  True,  True),
    ("rx 57",      "RDNA",    False, True),
    ("rx 56",      "RDNA",    False, True),
    ("rx 55",      "RDNA",    False, True),
    ("rx 54",      "RDNA",    False, True),
    ("radeon vii", "GCN 5",   False, True),
    ("vega",       "GCN 5",   False, True),
    ("rx 590",     "GCN 4",   False, True),
    ("rx 580",     "GCN 4",   False, True),
    ("rx 570",     "GCN 4",   False, True),
    ("rx 560",     "GCN 4",   False, True),
    ("rx 550",     "GCN 4",   False, True),
    ("rx 480",     "GCN 4",   False, True),
    ("rx 470",     "GCN 4",   False, True),
    ("rx 460",     "GCN 4",   False, True),
    ("instinct mi3", "CDNA 3", False, True),
    ("instinct mi2", "CDNA 2", False, True),
    ("instinct mi1", "CDNA",   False, True),
    ("radeon pro w7", "RDNA 3", True, True),
    ("radeon pro w6", "RDNA 2", True, True),
    ("radeon pro",    "GCN",   False, True),
]

# Memory type heuristics by architecture + bus width
_MEMORY_TYPE_HINTS: Dict[str, str] = {
    "Kepler":       "GDDR5",
    "Maxwell":      "GDDR5",
    "Pascal":       "GDDR5X",     # GP102/104 use GDDR5X; GP106/107 use GDDR5
    "Volta":        "HBM2",
    "Turing":       "GDDR6",
    "Ampere":       "GDDR6X",     # RTX 30xx; A100 uses HBM2e
    "Ada Lovelace": "GDDR6X",     # RTX 40xx
    "Hopper":       "HBM3",
    "Blackwell":    "GDDR7",      # RTX 50xx consumer; B100/B200 use HBM3e
    "GCN 4":        "GDDR5",
    "GCN 5":        "HBM2",       # Vega; RX 580 etc are GCN 4
    "RDNA":         "GDDR6",
    "RDNA 2":       "GDDR6",
    "RDNA 3":       "GDDR6",
    "RDNA 4":       "GDDR6",
    "CDNA":         "HBM2",
    "CDNA 2":       "HBM2e",
    "CDNA 3":       "HBM3",
}


def _classify_nvidia(name: str, major: int, minor: int, sms: int) -> Dict[str, Any]:
    """Derive NVIDIA architecture, core counts, and feature flags.

    Combines CUDA compute capability with device name to distinguish
    e.g. GTX 16xx (no tensor/RT) from RTX 20xx (tensor+RT) on same CC 7.5.
    """
    nl = name.lower()
    is_rtx = "rtx" in nl
    is_datacenter = any(k in nl for k in ("a100", "a800", "h100", "h200",
                                           "b100", "b200", "v100", "l40", "l4 "))
    arch = _NVIDIA_ARCH.get((major, minor))
    if arch is None:
        # Fallback: pick closest known arch (round down)
        for (maj, mi), info in sorted(_NVIDIA_ARCH.items(), reverse=True):
            if (maj, mi) <= (major, minor):
                arch = info
                break
    if arch is None:
        arch = ("Unknown", 64, 0, 0)

    arch_name, cores_per_sm, tensor_per_sm, rt_per_sm = arch
    cuda_cores = sms * cores_per_sm

    # ── Tensor cores ──
    has_tensor = False
    tensor_cores = 0
    if tensor_per_sm > 0:
        if is_rtx or is_datacenter or major >= 8:
            has_tensor = True
            tensor_cores = sms * tensor_per_sm
        elif major == 7 and minor == 0:
            # Volta (V100): always has tensor
            has_tensor = True
            tensor_cores = sms * tensor_per_sm

    # ── RT cores ──
    has_rt = False
    rt_cores = 0
    if rt_per_sm > 0:
        if is_rtx or (major >= 8 and minor >= 6):
            has_rt = True
            rt_cores = sms * rt_per_sm
        # Data-center chips (A100, H100): no RT cores
        if is_datacenter and major in (8, 9) and minor in (0,):
            has_rt = False
            rt_cores = 0

    # ── NVENC / NVDEC: Maxwell (sm5.0) and newer ──
    has_nvenc = major >= 5
    has_nvdec = major >= 3  # NVDEC since Kepler (VP5)

    # ── Memory type override for specific SKUs ──
    mem_type = _MEMORY_TYPE_HINTS.get(arch_name, "")
    if "a100" in nl:
        mem_type = "HBM2e"
    elif "h200" in nl:
        mem_type = "HBM3e"
    elif arch_name == "Pascal":
        # GP106/107 (GTX 1060/1050) use GDDR5, GP102/104 use GDDR5X
        if any(k in nl for k in ("1050", "1060", "1030")):
            mem_type = "GDDR5"
    elif arch_name == "Ampere" and not is_rtx:
        if is_datacenter:
            mem_type = "HBM2e"
    elif arch_name == "Ada Lovelace":
        if any(k in nl for k in ("4060", "4070")):
            mem_type = "GDDR6"  # lower-tier Ada uses GDDR6
    elif arch_name == "Blackwell" and is_datacenter:
        mem_type = "HBM3e"

    return {
        "architecture": arch_name,
        "cuda_capability": f"{major}.{minor}",
        "cuda_cores": cuda_cores,
        "tensor_cores": tensor_cores,
        "rt_cores": rt_cores,
        "has_tensor": has_tensor,
        "has_raytracing": has_rt,
        "has_nvenc": has_nvenc,
        "has_nvdec": has_nvdec,
        "memory_type": mem_type,
    }


def _classify_amd(name: str) -> Dict[str, Any]:
    """Derive AMD architecture and feature flags from device name."""
    nl = name.lower()
    for pattern, arch, has_ray, has_vcn in _AMD_ARCH_PATTERNS:
        if pattern in nl:
            return {
                "architecture": arch,
                "has_raytracing": has_ray,
                "has_nvenc": has_vcn,   # VCN encode mapped to has_nvenc flag
                "has_nvdec": has_vcn,   # VCN decode mapped to has_nvdec flag
                "memory_type": _MEMORY_TYPE_HINTS.get(arch, ""),
            }
    return {}


# ═══════════════════════════════════════════════════════════════════════════
#  Detection — enumerate all backends
# ═══════════════════════════════════════════════════════════════════════════

def detect_all() -> List[GPUDevice]:
    """Enumerate ALL GPUs from every available compute backend.

    Returns a list sorted by ``score`` (best first). Thread-safe, cached.
    """
    global _all_gpus, _best_gpu, _detected
    if _detected:
        return _all_gpus
    with _detect_lock:
        if _detected:
            return _all_gpus

        gpus: List[GPUDevice] = []
        seen: set[str] = set()

        # ── CuPy / CUDA ──
        gpus.extend(_probe_cuda(seen))

        # ── PyOpenCL ──
        gpus.extend(_probe_opencl(seen))

        # ── Intel oneAPI (dpctl) ──
        gpus.extend(_probe_intel(seen))

        # ── ARM / Android GPU ──
        gpus.extend(_probe_arm_gpu(seen))

        # ── SBC / IoT GPU (VideoCore, Tegra, Vivante, etc.) ──
        gpus.extend(_probe_sbc_gpu(seen))

        # ── Vulkan (Windows / Desktop — covers Intel Iris Xe, etc.) ──
        gpus.extend(_probe_vulkan(seen))

        # ── Enrich existing GPUs with shared VRAM & Vulkan info ──
        _enrich_shared_vram(gpus)

        # ── Enrich NVIDIA GPUs with nvidia-smi data (driver, power, PCIe) ──
        _enrich_nvidia_smi(gpus)

        # ── Enrich AMD GPUs with rocm-smi / architecture data ──
        _enrich_amd(gpus)

        # ── OS-level fallback ──
        if not gpus:
            for hw_name in _detect_os_gpu_names():
                vendor, discrete = _vendor_from_name(hw_name)
                gpus.append(GPUDevice(
                    name=hw_name, backend="none", vendor=vendor,
                    is_discrete=discrete,
                ))

        gpus.sort(key=lambda g: g.score, reverse=True)
        _all_gpus = gpus
        _best_gpu = gpus[0] if gpus else None

        if _best_gpu and _best_gpu.usable:
            log.info(
                "Best GPU: %s (%s, score=%d). Total: %d",
                _best_gpu.name, _best_gpu.backend, _best_gpu.score, len(gpus),
            )
        elif gpus:
            log.info("GPU(s) detected but no compute library: %s",
                     ", ".join(g.name for g in gpus))
        else:
            log.info("No GPU detected")

        _detected = True
        return _all_gpus


# ── Backend probes ──────────────────────────────────────────────────────

def _probe_cuda(seen: set[str]) -> List[GPUDevice]:
    """Probe CUDA devices via CuPy — extracts full hardware profile."""
    gpus: List[GPUDevice] = []
    try:
        import cupy as cp  # type: ignore[import-untyped]
        n = cp.cuda.runtime.getDeviceCount()
        for i in range(n):
            try:
                props = cp.cuda.runtime.getDeviceProperties(i)
                dev_name = props["name"]
                if isinstance(dev_name, bytes):
                    dev_name = dev_name.decode()
                mem = props.get("totalGlobalMem", 0)
                sms = props.get("multiProcessorCount", 0)
                major = props.get("major", 0)
                minor = props.get("minor", 0)
                vendor, discrete = _vendor_from_name(dev_name)

                # Clock / memory bus / cache / copy engines
                clock_khz = props.get("clockRate", 0)           # kHz
                mem_clock_khz = props.get("memoryClockRate", 0)  # kHz
                bus_width = props.get("memoryBusWidth", 0)       # bits
                l2_cache = props.get("l2CacheSize", 0)           # bytes
                async_engines = props.get("asyncEngineCount", 0)

                clock_mhz = clock_khz // 1000 if clock_khz else 0
                mem_clock_mhz = mem_clock_khz // 1000 if mem_clock_khz else 0
                # Effective bandwidth = 2 × mem_clock (DDR) × bus_width / 8
                bw_gbps = 0.0
                if mem_clock_mhz and bus_width:
                    bw_gbps = round(2 * mem_clock_mhz * bus_width / 8 / 1000, 1)

                # NVIDIA feature classification
                nv = _classify_nvidia(dev_name, major, minor, sms)

                gpu = GPUDevice(
                    name=dev_name, backend="cuda", vendor=vendor,
                    memory_bytes=mem, compute_units=sms,
                    is_discrete=discrete, _module=cp, _index=i,
                    architecture=nv.get("architecture", ""),
                    cuda_capability=nv.get("cuda_capability", ""),
                    cuda_cores=nv.get("cuda_cores", 0),
                    tensor_cores=nv.get("tensor_cores", 0),
                    rt_cores=nv.get("rt_cores", 0),
                    has_tensor=nv.get("has_tensor", False),
                    has_raytracing=nv.get("has_raytracing", False),
                    has_nvenc=nv.get("has_nvenc", False),
                    has_nvdec=nv.get("has_nvdec", False),
                    clock_mhz=clock_mhz,
                    boost_clock_mhz=clock_mhz,  # CUDA reports boost as clockRate
                    memory_clock_mhz=mem_clock_mhz,
                    memory_type=nv.get("memory_type", ""),
                    memory_bus_width=bus_width,
                    memory_bandwidth_gbps=bw_gbps,
                    l2_cache_bytes=l2_cache,
                    copy_engines=async_engines,
                )
                gpus.append(gpu)
                seen.add(dev_name.lower().strip())
            except Exception:
                pass
    except Exception as exc:
        log.debug("CuPy/CUDA not available: %s", exc)
    return gpus


def _probe_opencl(seen: set[str]) -> List[GPUDevice]:
    """Probe GPU devices via PyOpenCL — enriches AMD devices with arch info."""
    gpus: List[GPUDevice] = []
    try:
        import pyopencl as cl  # type: ignore[import-untyped]
        for plat in cl.get_platforms():
            try:
                for dev in plat.get_devices(device_type=cl.device_type.GPU):
                    dev_name = dev.name.strip()
                    key = dev_name.lower().strip()
                    if key in seen:
                        continue
                    vendor, discrete = _vendor_from_name(dev_name)
                    try:
                        cus = dev.max_compute_units
                    except Exception:
                        cus = 0

                    extra: Dict[str, Any] = {}
                    if vendor == "AMD":
                        extra = _classify_amd(dev_name)
                    elif vendor == "NVIDIA":
                        # CuPy should have caught this; but fill arch if not
                        extra = {"has_nvenc": True, "has_nvdec": True}

                    gpus.append(GPUDevice(
                        name=dev_name, backend="opencl", vendor=vendor,
                        memory_bytes=dev.global_mem_size,
                        compute_units=cus, is_discrete=discrete,
                        _module=cl, _index=0,
                        architecture=extra.get("architecture", ""),
                        has_raytracing=extra.get("has_raytracing", False),
                        has_nvenc=extra.get("has_nvenc", False),
                        has_nvdec=extra.get("has_nvdec", False),
                        memory_type=extra.get("memory_type", ""),
                    ))
                    seen.add(key)
            except Exception:
                continue
    except Exception as exc:
        log.debug("PyOpenCL not available: %s", exc)
    return gpus


def _probe_intel(seen: set[str]) -> List[GPUDevice]:
    """Probe GPU devices via Intel oneAPI (dpctl)."""
    gpus: List[GPUDevice] = []
    try:
        import dpctl  # type: ignore[import-untyped]
        for dev in dpctl.get_devices():
            if dev.device_type.name != "gpu":
                continue
            dev_name = dev.name
            key = dev_name.lower().strip()
            if key in seen:
                continue
            vendor, discrete = _vendor_from_name(dev_name)
            try:
                mem = dev.global_mem_size
            except Exception:
                mem = 0
            gpus.append(GPUDevice(
                name=dev_name, backend="intel", vendor=vendor,
                memory_bytes=mem, is_discrete=discrete,
                _module=dpctl, _index=0,
            ))
            seen.add(key)
    except Exception as exc:
        log.debug("dpctl/oneAPI not available: %s", exc)
    return gpus


# ── ARM / Android GPU probe ─────────────────────────────────────────────

def _probe_arm_gpu(seen: set[str]) -> List[GPUDevice]:
    """Detect ARM mobile / embedded GPUs (Adreno, Mali, Immortalis, etc.).

    Uses:
      1. SoC database from android.py (reliable)
      2. SBC / IoT database (Raspberry Pi VideoCore, Jetson Tegra, etc.)
      3. Android sysfs: /sys/class/kgsl (Adreno), /sys/class/misc/mali0 (Mali)
      4. Vulkan via ``vulkaninfo`` (if installed in Termux)
      5. OpenCL already catches ARM GPUs in _probe_opencl
    """
    gpus: List[GPUDevice] = []

    try:
        from pyaccelerate.android import is_android, is_arm, get_soc_info
    except ImportError:
        return gpus

    if not is_arm():
        return gpus

    # ── Try SoC database first ──
    soc = get_soc_info()
    if soc and soc.gpu_name:
        key = soc.gpu_name.lower().strip()
        if key not in seen:
            vendor, _ = _vendor_from_name(soc.gpu_name)
            gpu = GPUDevice(
                name=soc.gpu_name,
                backend="none",
                vendor=vendor,
                compute_units=soc.gpu_cores,
                is_discrete=False,
            )
            # Try OpenCL on ARM to upgrade backend
            gpu = _try_arm_opencl_upgrade(gpu) or gpu
            gpus.append(gpu)
            seen.add(key)
            return gpus  # SoC DB is authoritative

    # ── Sysfs fallback: KGSL (Qualcomm Adreno) ──
    try:
        from pathlib import Path
        kgsl_path = Path("/sys/class/kgsl/kgsl-3d0")
        if kgsl_path.exists():
            gpu_name = "Adreno GPU"
            try:
                gpu_model = (kgsl_path / "gpu_model").read_text().strip()
                if gpu_model:
                    gpu_name = f"Adreno {gpu_model}"
            except Exception:
                pass
            key = gpu_name.lower().strip()
            if key not in seen:
                gpus.append(GPUDevice(
                    name=gpu_name, backend="none", vendor="Qualcomm",
                    is_discrete=False,
                ))
                seen.add(key)
    except Exception:
        pass

    # ── Sysfs fallback: Mali ──
    try:
        from pathlib import Path
        mali_paths = [
            Path("/sys/class/misc/mali0"),
            Path("/sys/devices/platform/mali-midgard"),
            Path("/sys/module/mali_kbase"),
        ]
        for mp in mali_paths:
            if mp.exists():
                gpu_name = "Mali GPU"
                # Try to read GPU ID from kernel
                try:
                    p = Path("/sys/module/mali_kbase/parameters/gpu_id")
                    if p.exists():
                        gpu_name = f"Mali ({p.read_text().strip()})"
                except Exception:
                    pass
                key = gpu_name.lower().strip()
                if key not in seen:
                    gpus.append(GPUDevice(
                        name=gpu_name, backend="none", vendor="ARM",
                        is_discrete=False,
                    ))
                    seen.add(key)
                break
    except Exception:
        pass

    # ── Vulkan fallback (Termux) ──
    if not gpus:
        try:
            r = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    ll = line.lower()
                    if "devicename" in ll or "device name" in ll:
                        gpu_name = line.split("=", 1)[-1].strip().strip('"')
                        if ":" in gpu_name:
                            gpu_name = gpu_name.split(":", 1)[-1].strip()
                        key = gpu_name.lower().strip()
                        if key and key not in seen:
                            vendor, _ = _vendor_from_name(gpu_name)
                            gpus.append(GPUDevice(
                                name=gpu_name, backend="vulkan",
                                vendor=vendor, is_discrete=False,
                            ))
                            seen.add(key)
                        break
        except Exception:
            pass

    return gpus


def _probe_sbc_gpu(seen: set[str]) -> List[GPUDevice]:
    """Detect SBC / IoT GPUs (VideoCore, Tegra CUDA, Vivante, etc.).

    Sources:
      1. SBC database from iot.py (board model → SoC → GPU name)
      2. Jetson Tegra — already found via CUDA probe, but add fallback
      3. VideoCore (Raspberry Pi) via /dev/vchiq or vcgencmd
      4. Vivante (NXP i.MX) via /dev/galcore
    """
    gpus: List[GPUDevice] = []

    try:
        from pyaccelerate.iot import is_sbc, detect_sbc, is_jetson
    except ImportError:
        return gpus

    if not is_sbc():
        return gpus

    sbc = detect_sbc()
    if not sbc or not sbc.gpu_name:
        return gpus

    key = sbc.gpu_name.lower().strip()
    if key in seen:
        return gpus

    vendor = sbc.soc_vendor
    cuda_cores = sbc.gpu_cuda_cores
    backend = "none"

    # Jetson: CUDA is handled by _probe_cuda, but mark as cuda if available
    if sbc.family == "jetson" and cuda_cores > 0:
        try:
            import cupy  # type: ignore[import-untyped]
            backend = "cuda"
        except ImportError:
            pass

    # VideoCore: check for vcgencmd
    if "videocore" in key:
        vendor = "Broadcom"
        try:
            from pathlib import Path
            if Path("/dev/vchiq").exists():
                backend = "videocore"
        except Exception:
            pass

    # Vivante (NXP i.MX): check for galcore driver
    if "vivante" in key:
        vendor = "Vivante"
        try:
            from pathlib import Path
            if Path("/dev/galcore").exists():
                backend = "vivante"
        except Exception:
            pass

    gpu = GPUDevice(
        name=sbc.gpu_name,
        backend=backend,
        vendor=vendor,
        compute_units=cuda_cores,
        is_discrete=False,
    )
    # Try OpenCL upgrade
    gpu = _try_arm_opencl_upgrade(gpu) or gpu
    gpus.append(gpu)
    seen.add(key)

    return gpus


def _try_arm_opencl_upgrade(gpu: GPUDevice) -> Optional[GPUDevice]:
    """Try to upgrade an ARM GPU from 'none' to 'opencl' backend."""
    try:
        import pyopencl as cl  # type: ignore[import-untyped]
        for plat in cl.get_platforms():
            for dev in plat.get_devices(device_type=cl.device_type.GPU):
                dev_name = dev.name.strip()
                # Check if this CL device matches our GPU
                if _gpu_names_match(gpu.name, dev_name):
                    try:
                        cus = dev.max_compute_units
                    except Exception:
                        cus = gpu.compute_units
                    return GPUDevice(
                        name=dev_name,
                        backend="opencl",
                        vendor=gpu.vendor,
                        memory_bytes=dev.global_mem_size,
                        compute_units=cus,
                        is_discrete=False,
                        _module=cl,
                    )
    except Exception:
        pass
    return None


def _gpu_names_match(name_a: str, name_b: str) -> bool:
    """Check if two GPU name strings likely refer to the same device."""
    a, b = name_a.lower(), name_b.lower()
    # Exact substring match
    if a in b or b in a:
        return True
    # Check key identifiers (e.g., "adreno 750" vs "Qualcomm Adreno 750")
    for token in ("adreno", "mali", "immortalis", "xclipse", "powervr", "maleoon",
                   "videocore", "tegra", "vivante"):
        if token in a and token in b:
            return True
    return False


# ── Vulkan probe (Windows / Desktop) ────────────────────────────────────

def _probe_vulkan(seen: set[str]) -> List[GPUDevice]:
    """Detect Vulkan-capable GPUs via ``vulkaninfo``.

    This covers desktop Intel iGPUs (Iris Xe, UHD) with Vulkan support
    that lack CuPy, OpenCL or dpctl — the typical Ollama + Vulkan scenario.
    Also enriches *already-detected* GPUs with Vulkan version info by
    returning an empty list but annotating the ``seen`` set entries.
    """
    gpus: List[GPUDevice] = []
    output = _run_vulkaninfo()
    if not output:
        return gpus

    # Parse vulkaninfo output.  Supports both --summary and full formats:
    #   apiVersion     = 4202650 (1.2.154)   OR   apiVersion    = 1.2.154
    #   deviceName     = Intel(R) Iris(R) Xe Graphics
    current_name = ""
    current_api = ""
    for line in output.splitlines():
        stripped = line.strip()
        if "deviceName" in stripped and "=" in stripped:
            current_name = stripped.split("=", 1)[-1].strip()
        elif "apiVersion" in stripped and "=" in stripped:
            raw = stripped.split("=", 1)[-1].strip()
            # May be "4202650 (1.2.154)" or just "1.2.154"
            if "(" in raw:
                current_api = raw.split("(")[-1].rstrip(")")
            else:
                current_api = raw
        # When we have both, emit a device
        if current_name and current_api:
            key = current_name.lower().strip()
            vendor, discrete = _vendor_from_name(current_name)
            if key not in seen:
                mem, shared = _detect_vram_wmi(current_name)
                gpus.append(GPUDevice(
                    name=current_name,
                    backend="vulkan",
                    vendor=vendor,
                    memory_bytes=mem,
                    shared_memory_bytes=shared,
                    is_discrete=discrete,
                    vulkan_version=current_api,
                ))
                seen.add(key)
            current_name = ""
            current_api = ""

    if gpus:
        log.info("Vulkan probe found %d GPU(s): %s",
                 len(gpus), ", ".join(g.short_label() for g in gpus))
    return gpus


def _run_vulkaninfo() -> str:
    """Run vulkaninfo and return its stdout, or empty string on failure."""
    # Try --summary first (faster), fall back to full output
    for args in (["vulkaninfo", "--summary"], ["vulkaninfo"]):
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and "deviceName" in r.stdout:
                return r.stdout
        except FileNotFoundError:
            log.debug("vulkaninfo not found in PATH — Vulkan probe skipped")
            return ""
        except Exception:
            continue
    return ""


def _detect_vram_wmi(gpu_name_hint: str = "") -> Tuple[int, int]:
    """Detect dedicated and shared VRAM via WMI (Windows only).

    Returns ``(dedicated_bytes, shared_bytes)``.
    """
    if platform.system() != "Windows":
        return 0, 0
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name, AdapterRAM, "
             "AdapterDACType | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return 0, 0

        import json as _json
        data = _json.loads(r.stdout)
        if isinstance(data, dict):
            data = [data]

        hint = gpu_name_hint.lower()
        for ctrl in data:
            name = (ctrl.get("Name") or "").lower()
            if hint and hint not in name and name not in hint:
                continue
            adapter_ram = ctrl.get("AdapterRAM") or 0
            if isinstance(adapter_ram, str):
                adapter_ram = int(adapter_ram) if adapter_ram.isdigit() else 0
            # For Intel iGPU: AdapterRAM is typically the small dedicated
            # portion (128 MB).  Total usable VRAM is much larger (shared
            # from system RAM), reported by DxDiag.  We query it via
            # another CIM class if available.
            shared = 0
            if "intel" in name:
                shared = _detect_shared_vram_intel()
            return adapter_ram, shared
        return 0, 0
    except Exception as exc:
        log.debug("WMI VRAM detection failed: %s", exc)
        return 0, 0


def _detect_shared_vram_intel() -> int:
    """Detect Intel iGPU shared VRAM (system RAM allocated to GPU).

    On Windows, ``Win32_VideoController.AdapterRAM`` only reports the small
    dedicated portion (128 MB for Iris Xe).  The actual usable VRAM is
    shared from system RAM and reported by DxDiag as "Shared Memory".
    We approximate via available system RAM minus a safety margin.
    """
    try:
        import psutil  # type: ignore[import-untyped]
        total = psutil.virtual_memory().total
        # Intel iGPU can use up to half of system RAM (BIOS-dependent).
        # Conservative estimate: min(half of total RAM, 8 GB).
        shared = min(total // 2, 8 * 1024 ** 3)
        return shared
    except ImportError:
        return 0


def _enrich_shared_vram(gpus: List[GPUDevice]) -> None:
    """Post-process: add shared VRAM and Vulkan info for ALL GPU vendors.

    On Windows, every GPU (discrete or integrated) can use system RAM as
    shared GPU memory.  The Task Manager reports this for NVIDIA, AMD, and
    Intel alike — not just iGPUs.
    """
    if platform.system() != "Windows":
        return

    # Query WMI once for all adapters
    wmi_data = _query_wmi_all_adapters()

    for gpu in gpus:
        if gpu.shared_memory_bytes > 0:
            continue  # already enriched

        # Match WMI adapter by name
        matched = _match_wmi_adapter(gpu.name, wmi_data)
        if matched:
            dedicated = matched.get("dedicated", 0)
            shared = matched.get("shared", 0)
            if shared > 0:
                gpu.shared_memory_bytes = shared
            # If WMI reports better dedicated VRAM and we have none
            if dedicated > 0 and gpu.memory_bytes == 0:
                gpu.memory_bytes = dedicated
            # Driver version from WMI
            drv = matched.get("driver_version", "")
            if drv and not gpu.driver_version:
                gpu.driver_version = drv

        # Intel iGPU: fallback to estimated shared VRAM
        if gpu.vendor == "Intel" and not gpu.is_discrete and gpu.shared_memory_bytes == 0:
            gpu.shared_memory_bytes = _detect_shared_vram_intel()

        # Vulkan version if not set
        if not gpu.vulkan_version:
            gpu.vulkan_version = _detect_vulkan_version_for(gpu.name)


def _query_wmi_all_adapters() -> List[Dict[str, Any]]:
    """Query WMI for ALL video adapters with dedicated + shared memory."""
    if platform.system() != "Windows":
        return []
    try:
        # Use DxDiag-style query for shared memory too
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name,AdapterRAM,AdapterDACType,DriverVersion,"
             "CurrentHorizontalResolution | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return []
        import json as _json
        data = _json.loads(r.stdout)
        if isinstance(data, dict):
            data = [data]

        results = []
        for ctrl in data:
            name = ctrl.get("Name") or ""
            adapter_ram = ctrl.get("AdapterRAM") or 0
            if isinstance(adapter_ram, str):
                adapter_ram = int(adapter_ram) if adapter_ram.isdigit() else 0
            drv = ctrl.get("DriverVersion") or ""

            # Shared memory estimation: Windows allocates ~half system RAM
            shared = 0
            try:
                import psutil  # type: ignore[import-untyped]
                total_ram = psutil.virtual_memory().total
                # Windows typically allows up to half system RAM as shared GPU
                shared = min(total_ram // 2, 32 * 1024 ** 3)
            except ImportError:
                pass

            results.append({
                "name": name,
                "dedicated": adapter_ram,
                "shared": shared,
                "driver_version": drv,
            })
        return results
    except Exception as exc:
        log.debug("WMI adapter query failed: %s", exc)
        return []


def _match_wmi_adapter(gpu_name: str, wmi_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Match a GPUDevice name to a WMI adapter entry."""
    hint = gpu_name.lower()
    for adapter in wmi_data:
        wmi_name = (adapter.get("name") or "").lower()
        if hint in wmi_name or wmi_name in hint:
            return adapter
        # Token-based matching
        for tok in ("gtx", "rtx", "radeon", "rx", "arc", "iris", "uhd", "quadro"):
            if tok in hint and tok in wmi_name:
                # Check model number too
                import re
                nums_gpu = set(re.findall(r"\d{3,}", hint))
                nums_wmi = set(re.findall(r"\d{3,}", wmi_name))
                if nums_gpu and nums_gpu & nums_wmi:
                    return adapter
    return None


def _enrich_nvidia_smi(gpus: List[GPUDevice]) -> None:
    """Enrich NVIDIA GPUs with driver, power, PCIe, clocks from nvidia-smi."""
    nvidia_gpus = [g for g in gpus if g.vendor == "NVIDIA"]
    if not nvidia_gpus:
        return
    try:
        fields = ("index,name,driver_version,pcie.link.gen.max,"
                  "pcie.link.width.max,power.limit,"
                  "clocks.max.graphics,clocks.max.memory")
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return

        # Also grab CUDA driver version
        cuda_ver = ""
        try:
            r2 = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            # nvidia-smi header shows CUDA version
            r3 = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, text=True, timeout=10,
            )
            if r3.returncode == 0:
                for line in r3.stdout.splitlines():
                    if "CUDA Version" in line:
                        import re
                        m = re.search(r"CUDA Version:\s*([\d.]+)", line)
                        if m:
                            cuda_ver = m.group(1)
                        break
        except Exception:
            pass

        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            idx, name, driver, pcie_gen, pcie_w, power, boost, mem_clk = parts[:8]

            # Match to our GPUDevice
            for gpu in nvidia_gpus:
                if not _gpu_names_match(gpu.name, name):
                    continue
                if driver and driver != "[N/A]":
                    gpu.driver_version = driver
                if cuda_ver:
                    gpu.cuda_driver_version = cuda_ver
                try:
                    pg = int(pcie_gen)
                    if pg > 0:
                        gpu.pcie_gen = pg
                except (ValueError, TypeError):
                    pass
                try:
                    pw = int(pcie_w)
                    if pw > 0:
                        gpu.pcie_width = pw
                except (ValueError, TypeError):
                    pass
                try:
                    pl = int(float(power))
                    if pl > 0:
                        gpu.power_limit_w = pl
                except (ValueError, TypeError):
                    pass
                try:
                    bc = int(boost)
                    if bc > 0:
                        gpu.boost_clock_mhz = bc
                except (ValueError, TypeError):
                    pass
                try:
                    mc = int(mem_clk)
                    if mc > 0 and gpu.memory_clock_mhz == 0:
                        gpu.memory_clock_mhz = mc
                except (ValueError, TypeError):
                    pass
                break
    except FileNotFoundError:
        log.debug("nvidia-smi not found — NVIDIA enrichment skipped")
    except Exception as exc:
        log.debug("nvidia-smi enrichment failed: %s", exc)


def _enrich_amd(gpus: List[GPUDevice]) -> None:
    """Enrich AMD GPUs with architecture and rocm-smi data."""
    amd_gpus = [g for g in gpus if g.vendor == "AMD"]
    if not amd_gpus:
        return

    for gpu in amd_gpus:
        if not gpu.architecture:
            info = _classify_amd(gpu.name)
            if info:
                gpu.architecture = info.get("architecture", "")
                if not gpu.has_nvenc:
                    gpu.has_nvenc = info.get("has_nvenc", False)
                if not gpu.has_nvdec:
                    gpu.has_nvdec = info.get("has_nvdec", False)
                if not gpu.has_raytracing:
                    gpu.has_raytracing = info.get("has_raytracing", False)
                if not gpu.memory_type:
                    gpu.memory_type = info.get("memory_type", "")

    # Try rocm-smi for driver/clock info
    try:
        r = subprocess.run(
            ["rocm-smi", "--showdriverversion", "--showclocks", "--csv"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            for gpu in amd_gpus:
                for line in r.stdout.splitlines():
                    if "Driver version" in line:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            gpu.driver_version = parts[-1].strip()
    except (FileNotFoundError, Exception):
        pass

    # Windows fallback: WMI driver version
    if platform.system() == "Windows":
        for gpu in amd_gpus:
            if not gpu.driver_version:
                wmi_data = _query_wmi_all_adapters()
                matched = _match_wmi_adapter(gpu.name, wmi_data)
                if matched and matched.get("driver_version"):
                    gpu.driver_version = matched["driver_version"]


def _detect_vulkan_version_for(gpu_name: str) -> str:
    """Query vulkaninfo for a specific GPU's Vulkan API version."""
    output = _run_vulkaninfo()
    if not output:
        return ""
    hint = gpu_name.lower()
    current_api = ""
    for line in output.splitlines():
        stripped = line.strip()
        if "apiVersion" in stripped and "=" in stripped:
            raw = stripped.split("=", 1)[-1].strip()
            if "(" in raw:
                current_api = raw.split("(")[-1].rstrip(")")
            else:
                current_api = raw
        elif "deviceName" in stripped and "=" in stripped:
            name = stripped.split("=", 1)[-1].strip().lower()
            if hint in name or name in hint:
                return current_api
    return ""


# ── OS-level fallback ───────────────────────────────────────────────────

def _detect_os_gpu_names() -> List[str]:
    """Detect GPU names at OS level (no compute capability)."""
    names: List[str] = []
    try:
        if platform.system() == "Windows":
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().splitlines():
                    line = line.strip()
                    if line:
                        names.append(line)
        elif platform.system() == "Darwin":
            r = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if "Chipset Model" in line:
                        names.append(line.split(":", 1)[-1].strip())
        else:
            # ARM/Android: no lspci — use getprop or device-tree
            try:
                from pyaccelerate.android import is_android, get_soc_info
                if is_android():
                    soc = get_soc_info()
                    if soc and soc.gpu_name:
                        names.append(soc.gpu_name)
                    if not names:
                        # Try getprop for GPU hints
                        r2 = subprocess.run(
                            ["getprop", "ro.hardware.egl"],
                            capture_output=True, text=True, timeout=3,
                        )
                        if r2.returncode == 0 and r2.stdout.strip():
                            names.append(r2.stdout.strip())
            except ImportError:
                pass

            if not names:
                r = subprocess.run(
                    ["lspci"], capture_output=True, text=True, timeout=10,
                )
                for line in r.stdout.splitlines():
                    if "VGA" in line or "3D" in line or "Display" in line:
                        names.append(line.split(":", 2)[-1].strip())
    except Exception:
        pass
    return names


# ═══════════════════════════════════════════════════════════════════════════
#  Public convenience API
# ═══════════════════════════════════════════════════════════════════════════

def gpu_available() -> bool:
    """True if at least one GPU with a compute backend exists."""
    return any(g.usable for g in detect_all())


def best_gpu() -> Optional[GPUDevice]:
    """Return the highest-scored usable GPU, or None."""
    detect_all()
    if _best_gpu and _best_gpu.usable:
        return _best_gpu
    return None


def get_gpu_info() -> Dict[str, str]:
    """Info dict for the best GPU (or CPU fallback)."""
    gpus = detect_all()
    top = gpus[0] if gpus else None
    if top is None or not top.usable:
        hw = top.name if top else "N/A"
        return {
            "available": "false",
            "backend": "cpu",
            "device": hw or "N/A",
            "note": "No GPU compute library — using CPU",
        }
    return {
        "available": "true",
        "backend": top.backend,
        "device": top.name,
        "memory": f"{top.memory_gb:.1f} GB",
        "vendor": top.vendor,
        "score": str(top.score),
    }


def get_all_gpus_info() -> List[Dict[str, str]]:
    """Info dicts for ALL detected GPUs (best-first)."""
    return [g.as_dict() for g in detect_all()]


def get_install_hint() -> str:
    """Suggest pip install commands based on detected hardware."""
    gpus = detect_all()
    usable = [g for g in gpus if g.usable]
    if usable:
        return ""
    if not gpus:
        return "No GPU detected. CPU multi-threading will be used."
    hints: list[str] = []
    for g in gpus:
        vl = g.vendor.lower()
        if "nvidia" in vl:
            hints.append("pip install cupy-cuda12x")
        elif "intel" in vl:
            hints.append("pip install pyopencl")
        elif "amd" in vl:
            hints.append("pip install pyopencl")
    # ARM GPUs — suggest OpenCL or Vulkan
    for g in gpus:
        vl = g.vendor.lower()
        if vl in ("qualcomm", "arm", "samsung", "imagination", "hisilicon"):
            hints.append("pip install pyopencl  # ARM OpenCL")
    # SBC / IoT GPUs
    for g in gpus:
        vl = g.vendor.lower()
        nl = g.name.lower()
        if "broadcom" in vl or "videocore" in nl:
            hints.append("pip install pyopencl  # VideoCore OpenCL (RPi)")
        elif "vivante" in vl or "vivante" in nl:
            hints.append("pip install pyopencl  # Vivante OpenCL (i.MX)")
    if hints:
        return "Install GPU support:  " + "  or  ".join(sorted(set(hints)))
    return ""
