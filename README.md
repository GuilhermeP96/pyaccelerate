# PyAccelerate

**High-performance Python acceleration engine** — CPU, threads, virtual threads, multi-GPU, NPU, ARM/Android/Termux, OS priority, energy profiles and maximum optimization mode.

[![CI](https://github.com/GuilhermeP96/pyaccelerate/actions/workflows/ci.yml/badge.svg)](https://github.com/GuilhermeP96/pyaccelerate/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Features

| Module | Description |
|---|---|
| **`cpu`** | CPU detection, topology, NUMA, affinity, ISA flags, ARM big.LITTLE/DynamIQ, dynamic worker recommendations |
| **`threads`** | Persistent virtual-thread pool, sliding-window executor, async bridge, process pool |
| **`gpu`** | Multi-vendor GPU detection (NVIDIA/CUDA, AMD/OpenCL, Intel oneAPI, ARM Adreno/Mali/Immortalis), ranking, multi-GPU dispatch |
| **`npu`** | NPU detection & inference (OpenVINO, ONNX Runtime, DirectML, CoreML, ARM Hexagon/Samsung NPU/Tensor TPU/MediaTek APU) |
| **`virt`** | Virtualization detection (Hyper-V, VT-x/AMD-V, KVM, WSL2, Docker, container detection) |
| **`memory`** | Memory pressure monitoring, automatic worker clamping, reusable buffer pool |
| **`profiler`** | `@timed`, `@profile_memory` decorators, `Timer` context manager, `Tracker` statistics |
| **`benchmark`** | Built-in micro-benchmarks (CPU, threads, memory bandwidth, GPU compute) |
| **`priority`** | OS-level task priority (IDLE → REALTIME) & energy profiles (POWER_SAVER → ULTRA_PERFORMANCE) |
| **`max_mode`** | Maximum optimization mode — activates ALL resources simultaneously with OS tuning |
| **`android`** | Android/Termux platform detection, ARM SoC database (25+ chipsets), big.LITTLE, thermal & battery |
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
pyaccelerate gpu           # GPU details
pyaccelerate cpu           # CPU details
pyaccelerate npu           # NPU details
pyaccelerate android       # ARM/Android device details (SoC, clusters, thermal)
pyaccelerate virt          # Virtualization info
pyaccelerate memory        # Memory stats
pyaccelerate status        # One-liner
pyaccelerate priority      # Show current priority/energy
pyaccelerate priority --preset max     # Apply max performance preset
pyaccelerate priority --set high       # Set task priority
pyaccelerate priority --energy performance  # Set energy profile
pyaccelerate max-mode      # Show max-mode hardware manifest
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
├── engine.py       # Unified orchestrator
└── cli.py          # Command-line interface
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

- [ ] npm package (Node.js bindings via pybind11/napi)
- [ ] gRPC server mode for multi-language integration
- [ ] Kubernetes operator for auto-scaling GPU workloads
- [ ] Prometheus metrics exporter
- [ ] Auto-tuning feedback loop (benchmark → config → re-tune)

## Origin

Evolved from the acceleration & virtual-thread systems built for:
- [adb-toolkit](https://github.com/GuilhermeP96/adb-toolkit) — multi-GPU acceleration, virtual thread pool
- [python-gpu-statistical-analysis](https://github.com/GuilhermeP96/python-gpu-statistical-analysis) — GPU compute foundations

## License

MIT — see [LICENSE](LICENSE).
