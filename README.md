# PyAccelerate

**High-performance Python acceleration engine** — CPU, threads, virtual threads, multi-GPU, NPU, ARM/Android/Termux, IoT/SBC, auto-tuning, Prometheus metrics, HTTP/gRPC server, Kubernetes auto-scaling, and maximum optimization mode.

[![CI](https://github.com/GuilhermeP96/pyaccelerate/actions/workflows/ci.yml/badge.svg)](https://github.com/GuilhermeP96/pyaccelerate/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/pyaccelerate.svg)](https://pypi.org/project/pyaccelerate/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pyaccelerate.svg?label=PyPI%20downloads)](https://pypi.org/project/pyaccelerate/)
[![PyPI Stats](https://github.com/GuilhermeP96/pyaccelerate/actions/workflows/pypi-stats.yml/badge.svg)](https://github.com/GuilhermeP96/pyaccelerate/actions/workflows/pypi-stats.yml)

---

## Features

| Module | Description |
|---|---|
| **`cpu`** | CPU detection, topology, NUMA, affinity, ISA flags, ARM big.LITTLE/DynamIQ, dynamic worker recommendations |
| **`threads`** | Persistent virtual-thread pool, sliding-window executor, async bridge, process pool |
| **`work_stealing`** | Work-stealing scheduler (Tokio / Go runtime / ForkJoinPool style) — Chase-Lev deques, random victim selection |
| **`lockfree_queue`** | Lock-free MPMC queue & per-worker Chase-Lev deques for minimal contention |
| **`adaptive`** | Adaptive scheduler — auto-scales workers based on latency, CPU pressure & memory pressure |
| **`_native`** | Optional Cython / Rust (PyO3) accelerators for hot-path data structures |
| **`gpu`** | Multi-vendor GPU detection (NVIDIA/CUDA, AMD/OpenCL, Intel oneAPI, ARM Adreno/Mali/Immortalis), **architecture classification** (Kepler→Blackwell, GCN→RDNA4/CDNA3), **CUDA/Tensor/RT core counts**, NVENC/NVDEC & VCN encode/decode, clocks, memory type & bandwidth, PCIe info, driver version, shared VRAM for all vendors, Vulkan probing, ranking, multi-GPU dispatch |
| **`npu`** | NPU detection & inference (OpenVINO, ONNX Runtime, DirectML, CoreML, ARM Hexagon/Samsung NPU/Tensor TPU/MediaTek APU) |
| **`virt`** | Virtualization detection (Hyper-V, VT-x/AMD-V, KVM, WSL2, Docker, container detection) |
| **`memory`** | Memory pressure monitoring, automatic worker clamping, reusable buffer pool, **GPU memory stats (dedicated + shared VRAM, CUDA/Tensor/RT cores, bandwidth, features)** |
| **`profiler`** | `@timed`, `@profile_memory` decorators, `Timer` context manager, `Tracker` statistics |
| **`benchmark`** | Built-in micro-benchmarks (CPU, threads, memory bandwidth, GPU compute) |
| **`priority`** | OS-level task priority (IDLE → REALTIME) & energy profiles (POWER_SAVER → ULTRA_PERFORMANCE) |
| **`max_mode`** | Maximum optimization mode — activates ALL resources simultaneously with OS tuning |
| **`android`** | Android/Termux platform detection, ARM SoC database (25+ chipsets), big.LITTLE, thermal & battery |
| **`iot`** | IoT / SBC detection (Raspberry Pi, Jetson, BeagleBone, Coral, Hailo, 30+ SoCs) |
| **`autotune`** | Auto-tuning feedback loop — benchmark → config → re-tune, persistent profiles, **hardware-safe limits & config clamping**, GPU architecture/features/driver profiling |
| **`metrics`** | Prometheus metrics exporter (`/metrics` HTTP endpoint, all subsystems) |
| **`server`** | JSON HTTP & gRPC server for multi-language integration (Node.js, Go, Java, etc.) |
| **`k8s`** | Kubernetes operator — pod info, GPU node capacity, auto-scaling, manifest generation |
| **`engine`** | Unified orchestrator — auto-detects everything and provides a single API |

## Quick Start

```bash
pip install pyaccelerate
```

```python
from pyaccelerate import Engine

engine = Engine()
print(engine.summary())

# Submit I/O-bound tasks to the virtual thread pool
future = engine.submit(my_io_func, arg1, arg2)

# Run many tasks with auto-tuned concurrency
engine.run_parallel(process_file, [(f,) for f in files])

# GPU dispatch (auto-fallback to CPU)
results = engine.gpu_dispatch(my_kernel, data_chunks)
```

## Benchmark

> **"Ok... but how much faster is it?"** — Here are real numbers.

All benchmarks run on **Python 3.11 / Windows / 48-core Xeon** with `python -m benchmarks.run`.
Reproduce: `pip install pyaccelerate && python -m benchmarks.run`

### IO-bound — 200 simulated HTTP calls (20ms each)

| Runner | Time | Speedup | Tasks/sec |
|---|---|---|---|
| Sequential | 4.079s | 1.0× | 49 |
| `ThreadPoolExecutor` | 0.118s | 34.7× | 1,701 |
| **`pyaccelerate.engine`** | **0.193s** | **21.2×** | **1,038** |
| **`pyaccelerate.threads`** | **0.150s** | **27.3×** | **1,336** |
| **`pyaccelerate.ws`** | **0.240s** | **17.0×** | **832** |

### IO-bound — 200 variable-latency calls (5–80ms, realistic)

| Runner | Time | Speedup | Tasks/sec |
|---|---|---|---|
| Sequential | 8.350s | 1.0× | 24 |
| `ThreadPoolExecutor` | 0.245s | 34.1× | 817 |
| **`pyaccelerate.threads`** | **0.329s** | **25.4×** | **608** |
| **`pyaccelerate.ws`** | **0.369s** | **22.6×** | **542** |
| **`pyaccelerate.adaptive`** | **0.656s** | **12.7×** | **305** |

### CPU-bound — zlib compress 200KB × 100 (GIL released by C extension)

| Runner | Time | Speedup | Tasks/sec |
|---|---|---|---|
| Sequential | 0.153s | 1.0× | 654 |
| `ThreadPoolExecutor` | 0.049s | 3.1× | 2,026 |
| **`pyaccelerate.threads`** | **0.049s** | **3.1×** | **2,033** |
| **`pyaccelerate.ws`** | **0.079s** | **1.9×** | **1,264** |

### Mixed — IO (10ms) + CPU (SHA-256 × 400) × 200

| Runner | Time | Speedup | Tasks/sec |
|---|---|---|---|
| Sequential | 2.216s | 1.0× | 90 |
| `ThreadPoolExecutor` | 0.197s | 11.3× | 1,017 |
| **`pyaccelerate.threads`** | **0.162s** | **13.7×** | **1,233** |
| **`pyaccelerate.engine`** | **0.206s** | **10.8×** | **972** |
| **`pyaccelerate.adaptive`** | **0.634s** | **3.5×** | **315** |

> **Key takeaway:** `pyaccelerate.threads` beats `ThreadPoolExecutor` by **21%** on mixed workloads.
> The work-stealing scheduler (`ws`) excels at **variable-latency IO** where load balancing matters most.
> Run your own benchmarks: `python -m benchmarks.run` (full) or `python -m benchmarks.run --quick` (CI).

---

## Maximum Optimization Mode

Activates **all** available hardware resources in parallel with OS-level tuning:

```python
from pyaccelerate.max_mode import MaxMode

with MaxMode() as m:
    print(m.summary())  # hardware manifest

    # Run CPU + I/O simultaneously
    results = m.run_all(
        cpu_fn=cpu_heavy_task, cpu_items=cpu_data,
        io_fn=io_heavy_task, io_items=io_data,
    )

    # I/O only (thread pool)
    downloaded = m.run_io(download, [(url,) for url in urls])

    # CPU only (process pool)
    computed = m.run_cpu(crunch, [(n,) for n in numbers])

    # Multi-stage pipeline
    results = m.run_pipeline([
        ("download", download_fn, urls),
        ("transform", transform_fn, data),
        ("save", save_fn, output),
    ])
```

Or via the Engine:

```python
engine = Engine()
with engine.max_mode() as m:
    results = m.run_all(...)
```

## OS Priority & Energy Management

Control process scheduling and power profiles across Windows, Linux & macOS:

```python
from pyaccelerate.priority import (
    TaskPriority, EnergyProfile,
    set_task_priority, set_energy_profile,
    max_performance, balanced, power_saver,
)

# Quick presets
max_performance()   # HIGH priority + ULTRA_PERFORMANCE energy
balanced()          # Restore defaults
power_saver()       # BELOW_NORMAL + POWER_SAVER

# Fine-grained control
set_task_priority(TaskPriority.ABOVE_NORMAL)
set_energy_profile(EnergyProfile.PERFORMANCE)
```

## CLI

```bash
pyaccelerate info          # Full hardware report
pyaccelerate benchmark     # Run micro-benchmarks
pyaccelerate gpu           # GPU details (architecture, cores, VRAM, clocks, features, driver, PCIe)
pyaccelerate cpu           # CPU details
pyaccelerate npu           # NPU details
pyaccelerate android       # ARM/Android device details (SoC, clusters, thermal)
pyaccelerate virt          # Virtualization info
pyaccelerate memory        # Memory stats (system + GPU dedicated/shared)
pyaccelerate limits        # Hardware-safe batch/worker limits
pyaccelerate limits --json # Limits as JSON
pyaccelerate status        # One-liner
pyaccelerate priority      # Show current priority/energy
pyaccelerate priority --preset max     # Apply max performance preset
pyaccelerate priority --set high       # Set task priority
pyaccelerate priority --energy performance  # Set energy profile
pyaccelerate max-mode      # Show max-mode hardware manifest
pyaccelerate tune          # Auto-tune: benchmark → optimise → save
pyaccelerate tune --apply  # Tune and apply to current process
pyaccelerate tune --show   # Show current tune profile
pyaccelerate metrics       # Start Prometheus /metrics server (:9090)
pyaccelerate metrics --once# Print metrics and exit
pyaccelerate serve         # Start HTTP/gRPC API server (:8420)
pyaccelerate k8s           # Kubernetes pod & GPU info
pyaccelerate k8s --manifest# Generate K8s Deployment YAML
pyaccelerate iot           # IoT / SBC board details
pyaccelerate version       # Print version
```

## ARM / Android / Termux Support

Full hardware detection for ARM devices — phones (Termux, Pydroid), tablets, Raspberry Pi, ARM laptops (Snapdragon X Elite), and ARM servers:

```python
from pyaccelerate.android import (
    is_android, is_termux, is_arm,
    get_device_info, get_soc_info,
    detect_big_little, get_arm_features,
    get_thermal_zones, get_battery_info,
)

if is_arm():
    soc = get_soc_info()
    if soc:
        print(f"{soc.name} ({soc.vendor})")   # Snapdragon 8 Gen 3 (Qualcomm)
        print(f"GPU: {soc.gpu_name}")           # Adreno 750
        print(f"NPU: {soc.npu_name} ({soc.npu_tops} TOPS)")  # Hexagon NPU (73.0 TOPS)

    clusters = detect_big_little()
    # {"Cortex-X4": [0], "Cortex-A720": [1,2,3], "Cortex-A520": [4,5,6,7]}

    features = get_arm_features()
    # ["aes", "asimd", "bf16", "crc32", "neon", "sve", "sve2", ...]
```

**Supported SoC families** (25+ chipsets in database):
- **Qualcomm** — Snapdragon 8 Elite, 8/7/6 Gen 1-3, 888, 865, X Elite
- **Samsung** — Exynos 2500, 2200, 2100, 1380, 990
- **Google** — Tensor G1–G4
- **MediaTek** — Dimensity 9300, 9200, 9000, 8300, 1200, 1100, 900
- **HiSilicon** — Kirin 9010, 9000
- **Unisoc** — T616

**ARM GPU detection** — Adreno, Mali, Immortalis, Xclipse, PowerVR, Maleoon (via SoC DB, sysfs, Vulkan, OpenCL)

**ARM NPU detection** — Hexagon, Samsung NPU, Google TPU, MediaTek APU, Da Vinci NPU (via SoC DB, NNAPI, TFLite)

## Modules in Depth

### Virtual Thread Pool

Inspired by Java's virtual threads — a persistent `ThreadPoolExecutor` sized for I/O (`cores × 3`, cap 32). All I/O-bound work shares this pool instead of creating/destroying threads per operation.

```python
from pyaccelerate.threads import get_pool, run_parallel, submit

# Single task
fut = submit(download_file, url)

# Bounded concurrency (sliding window)
run_parallel(process, [(item,) for item in items], max_concurrent=8)
```

### Work-Stealing Scheduler

High-performance scheduler inspired by **Tokio** (Rust), **Go runtime** and **Java ForkJoinPool**.
Each worker owns a local Chase-Lev deque — pops LIFO (cache-friendly), steals FIFO (fair). When idle, workers steal from random victims with exponential back-off parking.

```python
from pyaccelerate.work_stealing import WorkStealingScheduler, ws_submit, ws_map

# Module-level convenience
fut = ws_submit(my_func, arg1, arg2)
results = ws_map(fn, [(a,), (b,), (c,)])

# Full control
with WorkStealingScheduler(num_workers=8, steal_batch_size=4) as sched:
    futures = [sched.submit(process, item) for item in items]
    results = [f.result() for f in futures]
    print(sched.stats())  # completed, stolen, avg_latency_us
```

Or via the Engine:

```python
engine = Engine()
fut = engine.ws_submit(my_func, arg1)
results = engine.ws_map(fn, [(a,), (b,)])
```

### Lock-Free Task Queue

The `lockfree_queue` module provides two data structures underlying the work-stealing scheduler:

- **`WorkDeque`**: Per-worker Chase-Lev deque — owner push/pop lock-free (GIL + `collections.deque`), stealers use a lightweight spinlock.
- **`MPMCQueue`**: Multi-Producer Multi-Consumer global injection queue with efficient `Event`-based parking.

```python
from pyaccelerate.lockfree_queue import WorkDeque, MPMCQueue

# Per-worker deque
d = WorkDeque()
d.push(task)
task = d.pop()        # LIFO (owner)
task = d.steal()      # FIFO (other workers)
batch = d.steal_batch(4)

# Global queue
q = MPMCQueue()
q.put(task)
q.put_batch([t1, t2, t3])
task = q.get()
q.wait(timeout=1.0)   # block until items arrive
```

### Adaptive Scheduler

Wraps the work-stealing scheduler and **dynamically tunes** worker count based on real-time metrics:

| Signal | Action |
|---|---|
| P95 latency > threshold + CPU < 70% | Scale **up** workers |
| CPU utilisation > 90% | Scale **down** workers |
| Memory pressure HIGH/CRITICAL | Shed workers immediately |
| P95 latency very low (idle) | Scale **down** to save resources |
| CPU load changes | Auto-tune steal batch size |

```python
from pyaccelerate.adaptive import AdaptiveScheduler, AdaptiveConfig

cfg = AdaptiveConfig(
    min_workers=2,
    max_workers=16,
    cooldown_seconds=2.0,
)

with AdaptiveScheduler(config=cfg) as sched:
    results = sched.map(process, [(item,) for item in data])
    print(sched.snapshot())  # workers, p95, cpu%, mem_pressure, adjustments
```

Or via the Engine:

```python
engine = Engine()
with engine.adaptive_scheduler() as sched:
    results = sched.map(heavy_fn, items)
```

### Native Accelerators (Optional)

For maximum throughput, compile the hot-path data structures to native code:

**Cython** (C extension):
```bash
pip install cython
cd src/pyaccelerate/_native
python setup_cython.py build_ext --inplace
```

**Rust** (PyO3 + crossbeam-deque — same algorithm as Tokio):
```bash
cd bindings/rust/pyaccelerate_native
pip install maturin
maturin develop --release
```

When a native extension is installed, it's used automatically — no code changes needed. The pure-Python fallback is always available.

### Multi-GPU Dispatch

Auto-detects GPUs across CUDA, OpenCL and Intel oneAPI. Distributes workloads with configurable strategies.

```python
from pyaccelerate.gpu import detect_all, dispatch

gpus = detect_all()
results = dispatch(my_kernel, data_chunks, strategy="score-weighted")
```

### Profiling

Zero-config decorators for timing and memory tracking:

```python
from pyaccelerate.profiler import timed, profile_memory, Tracker

@timed(level=logging.INFO)
def heavy_computation():
    ...

tracker = Tracker("db_queries")
for batch in batches:
    with tracker.measure():
        run_query(batch)
print(tracker.summary())
```

### Auto-Tuning Feedback Loop

Benchmark your hardware, persist the optimal configuration, and auto-apply it:

```python
from pyaccelerate.autotune import auto_tune, get_or_tune, apply_profile

# Run a full tune cycle (benchmark → save to ~/.pyaccelerate/)
profile = auto_tune()
print(f"Overall score: {profile.overall_score}/100")
print(f"Optimal IO workers: {profile.optimal_io_workers}")
print(f"Optimal CPU workers: {profile.optimal_cpu_workers}")

# Load existing or re-tune if hardware changed / profile stale
profile = get_or_tune()

# Apply to running process (sets workers, priority, energy)
apply_profile()
```

### Hardware-Safe Configuration

Prevent runaway concurrency on constrained hardware (iGPUs, LLM inference, Ollama, etc.):

```python
from pyaccelerate.autotune import hardware_safe_limits, clamp_config

# Get hardware-derived upper bounds
limits = hardware_safe_limits()
# Discrete GPU:  {"batch_size": 32, "max_workers": N, "max_concurrent": 4}
# Integrated GPU: {"batch_size": 8, "max_workers": 1, "max_concurrent": 2}
# CPU-only:      {"batch_size": 5, "max_workers": 1, "max_concurrent": 2}

# Clamp user/config values to safe maximums
user_config = {"batch_size": 50, "max_workers": 4}
safe = clamp_config(user_config)
# iGPU → {"batch_size": 8, "max_workers": 1}
```

### GPU Memory Stats

Query dedicated + shared VRAM and hardware details for the best GPU:

```python
from pyaccelerate.memory import get_gpu_memory_stats

stats = get_gpu_memory_stats()
# {"gpu_available": 1.0, "gpu_dedicated_gb": 6.0,
#  "gpu_shared_gb": 28.0, "gpu_total_gb": 34.0,
#  "gpu_is_discrete": 1.0, "gpu_vulkan": 1.0,
#  "gpu_cuda_cores": 1408, "gpu_tensor_cores": 0,
#  "gpu_has_hw_encode": 1.0, "gpu_has_hw_decode": 1.0,
#  "gpu_memory_bandwidth_gbps": 336.0, "gpu_boost_clock_mhz": 2130}
```

### GPU Shared VRAM & Vulkan Detection

The `GPUDevice` dataclass exposes comprehensive GPU hardware details — architecture, cores, features, clocks, memory type, driver, PCIe, and shared VRAM:

```python
from pyaccelerate.gpu import detect_all

for g in detect_all():
    print(f"{g.name}: {g.memory_gb:.1f} GB dedicated")
    print(f"  Architecture:    {g.architecture} | CC {g.cuda_capability}")
    print(f"  Cores:           CUDA={g.cuda_cores} Tensor={g.tensor_cores} RT={g.rt_cores}")
    print(f"  Shared VRAM:     {g.shared_memory_gb:.1f} GB")
    print(f"  Total:           {g.total_memory_gb:.1f} GB")
    print(f"  Memory:          {g.memory_type} | {g.memory_bus_width}-bit | {g.memory_bandwidth_gbps:.0f} GB/s")
    print(f"  Clock:           {g.boost_clock_mhz} MHz boost")
    print(f"  Features:        {', '.join(g.features)}")
    print(f"  HW Encode/Dec:   NVENC={g.has_nvenc} NVDEC={g.has_nvdec}")
    print(f"  PCIe:            Gen{g.pcie_gen} x{g.pcie_width}")
    print(f"  Driver:          {g.driver_version} | CUDA: {g.cuda_driver_version}")
    print(f"  Vulkan:          {g.vulkan_version or 'not detected'}")
```

#### Supported GPU Attributes

| Field | Type | Description |
|---|---|---|
| `architecture` | `str` | GPU architecture (Turing, Ampere, Ada Lovelace, RDNA 3, etc.) |
| `cuda_capability` | `str` | NVIDIA Compute Capability (e.g. "7.5") |
| `cuda_cores` | `int` | Total CUDA cores (NVIDIA) or Stream Processors (AMD) |
| `tensor_cores` | `int` | Tensor core count (RTX / data-center only) |
| `rt_cores` | `int` | RT core count (RTX only) |
| `has_tensor` | `bool` | Tensor / mixed-precision acceleration available |
| `has_raytracing` | `bool` | Hardware ray tracing available |
| `has_nvenc` | `bool` | Hardware video encoder (NVENC / VCN) |
| `has_nvdec` | `bool` | Hardware video decoder (NVDEC / VCN) |
| `clock_mhz` | `int` | Base clock (MHz) |
| `boost_clock_mhz` | `int` | Boost clock (MHz) |
| `memory_clock_mhz` | `int` | Memory clock (MHz) |
| `memory_type` | `str` | Memory type (GDDR5, GDDR6, GDDR6X, HBM2e, HBM3) |
| `memory_bus_width` | `int` | Memory bus width (bits) |
| `memory_bandwidth_gbps` | `float` | Effective memory bandwidth (GB/s) |
| `pcie_gen` | `int` | PCIe generation (3, 4, 5) |
| `pcie_width` | `int` | PCIe link width (16, 8, etc.) |
| `driver_version` | `str` | GPU driver version |
| `cuda_driver_version` | `str` | CUDA runtime version |
| `power_limit_w` | `int` | TDP / power limit (watts) |
| `copy_engines` | `int` | Async DMA copy engines |
| `features` | `list[str]` | Capability flags (compute, tensor, hw_encode, cuda_7.5, turing, etc.) |
| `shared_memory_bytes` | `int` | Shared system memory (all vendors on Windows) |
| `vulkan_version` | `str` | Vulkan API version |

#### NVIDIA Architecture Coverage

Kepler (3.0) → Maxwell (5.x) → Pascal (6.x) → Volta (7.0) → Turing (7.5) → Ampere (8.x) → Ada Lovelace (8.9) → Hopper (9.0) → Blackwell (10.x)

- **GTX 16xx** (Turing): CUDA cores + NVENC/NVDEC, no Tensor/RT cores
- **RTX 20xx** (Turing): Full Tensor + RT cores
- **RTX 30xx** (Ampere): Enhanced Tensor + RT cores, 2nd gen
- **RTX 40xx** (Ada Lovelace): 4th gen Tensor, 3rd gen RT, DLSS 3
- **RTX 50xx** (Blackwell): 5th gen Tensor, 4th gen RT
- **Data center** (A100/H100/B100): Tensor cores, no RT cores

#### AMD Architecture Coverage

GCN 4 (RX 400/500) → GCN 5 (Vega) → RDNA (RX 5000) → RDNA 2 (RX 6000) → RDNA 3 (RX 7000) → RDNA 4 (RX 9000) → CDNA/CDNA 2/CDNA 3 (Instinct MI series)

### Prometheus Metrics

Expose CPU/GPU/NPU/memory/pool metrics in Prometheus format:

```python
from pyaccelerate.metrics import start_metrics_server, get_metrics_text

# Start /metrics endpoint on port 9090
start_metrics_server(port=9090)

# Or get text for your own framework
text = get_metrics_text()
```

```bash
pyaccelerate metrics --port 9090     # Start server
pyaccelerate metrics --once          # Print and exit
curl http://localhost:9090/metrics    # Scrape
```

### HTTP / gRPC Server

Multi-language access to all PyAccelerate features:

```python
from pyaccelerate.server import PyAccelerateServer

with PyAccelerateServer(http_port=8420, grpc_port=50051) as srv:
    print(f"HTTP: {srv.http_url}/api/v1")
    # Block until Ctrl+C
    srv.start(block=True)
```

```bash
pyaccelerate serve --http-port 8420 --grpc-port 50051
curl http://localhost:8420/api/v1/info    # JSON
curl http://localhost:8420/api/v1/cpu
curl http://localhost:8420/api/v1/gpu
curl http://localhost:8420/api/v1/metrics  # Prometheus text
```

### Kubernetes Integration

Pod detection, GPU node capacity, auto-scaling recommendations & manifest generation:

```python
from pyaccelerate.k8s import (
    is_kubernetes, get_pod_info,
    get_scaling_recommendation, generate_resource_manifest,
)

if is_kubernetes():
    pod = get_pod_info()
    print(f"Pod: {pod.name} | GPU: {pod.gpu_limit}")

rec = get_scaling_recommendation()
print(f"Replicas: {rec.recommended_replicas} ({rec.reason})")

yaml = generate_resource_manifest(name="ml-worker", gpu_per_replica=1)
```

```bash
pyaccelerate k8s                # Show pod & GPU info
pyaccelerate k8s --manifest     # Generate Deployment YAML
pyaccelerate k8s --json         # Machine-readable
```

### Node.js / npm Client

A zero-dependency Node.js client is included in `bindings/nodejs/`:

```javascript
const { PyAccelerate } = require('pyaccelerate');

const client = new PyAccelerate('http://localhost:8420');
const info = await client.getInfo();
const metrics = await client.getMetrics();
const bench = await client.runBenchmark();
```

## Installation Options

```bash
# Core (CPU + threads + memory + virt)
pip install pyaccelerate

# With NVIDIA GPU support
pip install pyaccelerate[cuda]

# With OpenCL support (AMD/Intel/NVIDIA)
pip install pyaccelerate[opencl]

# With Intel oneAPI support
pip install pyaccelerate[intel]

# All GPU backends
pip install pyaccelerate[all-gpu]

# gRPC server mode
pip install pyaccelerate[grpc]

# Kubernetes integration
pip install pyaccelerate[k8s]

# Development
pip install pyaccelerate[dev]
```

## Docker

```bash
# CPU-only
docker build -t pyaccelerate .
docker run --rm pyaccelerate info

# With NVIDIA GPU
docker build -f Dockerfile.gpu -t pyaccelerate:gpu .
docker run --rm --gpus all pyaccelerate:gpu info

# Docker Compose
docker compose up pyaccelerate    # CPU
docker compose up gpu             # GPU
```

## Development

```bash
git clone https://github.com/GuilhermeP96/pyaccelerate.git
cd pyaccelerate
pip install -e ".[dev]"

# Run tests
pytest -v

# Run benchmarks
python -m benchmarks.run            # full suite
python -m benchmarks.run --quick    # CI-friendly (fewer tasks)
python -m benchmarks.run --io       # IO-bound only
python -m benchmarks.run --cpu      # CPU-bound only
python -m benchmarks.run --mixed    # mixed workloads only

# Lint + format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/

# Build wheel
python -m build
```

## Architecture

```
pyaccelerate/
├── cpu.py          # CPU detection & topology
├── threads.py      # Virtual thread pool & executors
├── gpu/
│   ├── detector.py # Multi-vendor GPU enumeration
│   ├── cuda.py     # CUDA/CuPy helpers
│   ├── opencl.py   # PyOpenCL helpers
│   ├── intel.py    # Intel oneAPI helpers
│   └── dispatch.py # Multi-GPU load balancer
├── npu/
│   ├── detector.py # NPU detection (Intel, Qualcomm, Apple)
│   ├── onnx_rt.py  # ONNX Runtime inference
│   ├── openvino.py # OpenVINO inference
│   └── inference.py# Unified inference API
├── virt.py         # Virtualization detection
├── memory.py       # Memory monitoring & buffer pool
├── profiler.py     # Timing & profiling utilities
├── benchmark.py    # Built-in micro-benchmarks
├── priority.py     # OS task priority & energy profiles
├── max_mode.py     # Maximum optimization mode
├── iot.py          # IoT / SBC hardware detection
├── autotune.py     # Auto-tuning feedback loop
├── metrics.py      # Prometheus metrics exporter
├── server.py       # HTTP + gRPC multi-language API
├── k8s.py          # Kubernetes pod & GPU integration
├── lockfree_queue.py # Lock-free MPMC & Chase-Lev deques
├── work_stealing.py  # Work-stealing scheduler (Tokio/Go/FJP)
├── adaptive.py       # Adaptive pressure-driven scheduler
├── engine.py         # Unified orchestrator
├── cli.py            # Command-line interface
├── _native/          # Optional Cython accelerators
│   ├── _fast_deque.pyx
│   └── setup_cython.py
└── bindings/
    ├── nodejs/       # npm client for Node.js / TypeScript
    └── rust/         # PyO3 Rust native extension
        └── pyaccelerate_native/
```

## Examples

The `examples/` directory contains runnable scripts demonstrating all features:

| Example | Description |
|---|---|
| `example_basic.py` | Engine creation, summary, submit, run_parallel, batch |
| `example_parallel_io.py` | Parallel download/process/write with public UCI ML datasets |
| `example_cpu_bound.py` | Sequential vs thread pool vs process pool comparison |
| `example_max_mode.py` | MaxMode context manager, run_all, run_io, run_cpu, pipeline |
| `example_pipeline.py` | Multi-stage data pipeline (download → analyze → report) |
| `example_priority.py` | TaskPriority levels, EnergyProfile, presets, benchmarking |

```bash
cd examples
python example_basic.py
python example_max_mode.py
python example_priority.py
```

## Roadmap

- [x] IoT / SBC detection (Raspberry Pi, Jetson, Coral, Hailo)
- [x] Auto-tuning feedback loop (benchmark → config → re-tune)
- [x] Prometheus metrics exporter
- [x] gRPC server mode for multi-language integration
- [x] Kubernetes operator for auto-scaling GPU workloads
- [x] npm package (Node.js bindings via HTTP API)
- [x] Work-stealing scheduler (Tokio / Go / ForkJoinPool style)
- [x] Lock-free task queues (Chase-Lev deques, MPMC)
- [x] Adaptive scheduler (latency, CPU & memory pressure)
- [x] Optional native accelerators (Cython + Rust/PyO3)
- [x] Benchmark suite with IO/CPU/mixed workload comparisons
- [x] Shared VRAM detection for integrated GPUs (Intel Iris Xe, etc.)
- [x] Vulkan version probing per GPU
- [x] Hardware-safe configuration limits (`hardware_safe_limits`, `clamp_config`)
- [x] GPU memory stats API (`get_gpu_memory_stats`)

## Origin

Evolved from the acceleration & virtual-thread systems built for:
- [adb-toolkit](https://github.com/GuilhermeP96/adb-toolkit) — multi-GPU acceleration, virtual thread pool
- [python-gpu-statistical-analysis](https://github.com/GuilhermeP96/python-gpu-statistical-analysis) — GPU compute foundations

## License

MIT — see [LICENSE](LICENSE).

<!-- PYPI-STATS-START -->
## 📊 PyPI Download Statistics

> Last updated: 2026-03-24 01:06 UTC (refreshed every Monday via GitHub Actions — [run manually](https://github.com/GuilhermeP96/pyaccelerate/actions/workflows/pypi-stats.yml))

| Period      | Downloads |
|-------------|-----------|
| Last day    | 2 |
| Last week   | 146 |
| Last month  | 1,111 |
| **Total**   | **3,534** |

### By Python version (top 5)

| Version | Downloads |
|---------|-----------|
| Python 3 | 158 |
<!-- PYPI-STATS-END -->
