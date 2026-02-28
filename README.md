# PyAccelerate

**High-performance Python acceleration engine** — CPU, threads, virtual threads, multi-GPU and virtualization.

[![CI](https://github.com/GuilhermeP96/pyaccelerate/actions/workflows/ci.yml/badge.svg)](https://github.com/GuilhermeP96/pyaccelerate/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Features

| Module | Description |
|---|---|
| **`cpu`** | CPU detection, topology, NUMA, affinity, ISA flags, dynamic worker recommendations |
| **`threads`** | Persistent virtual-thread pool, sliding-window executor, async bridge, process pool |
| **`gpu`** | Multi-vendor GPU detection (NVIDIA/CUDA, AMD/OpenCL, Intel oneAPI), ranking, multi-GPU dispatch |
| **`virt`** | Virtualization detection (Hyper-V, VT-x/AMD-V, KVM, WSL2, Docker, container detection) |
| **`memory`** | Memory pressure monitoring, automatic worker clamping, reusable buffer pool |
| **`profiler`** | `@timed`, `@profile_memory` decorators, `Timer` context manager, `Tracker` statistics |
| **`benchmark`** | Built-in micro-benchmarks (CPU, threads, memory bandwidth, GPU compute) |
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

## CLI

```bash
pyaccelerate info          # Full hardware report
pyaccelerate benchmark     # Run micro-benchmarks
pyaccelerate gpu           # GPU details
pyaccelerate cpu           # CPU details
pyaccelerate virt          # Virtualization info
pyaccelerate memory        # Memory stats
pyaccelerate status        # One-liner
```

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
├── virt.py         # Virtualization detection
├── memory.py       # Memory monitoring & buffer pool
├── profiler.py     # Timing & profiling utilities
├── benchmark.py    # Built-in micro-benchmarks
├── engine.py       # Unified orchestrator
└── cli.py          # Command-line interface
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
