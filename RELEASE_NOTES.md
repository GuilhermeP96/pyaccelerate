## What's New

### 0.11.1 — Machine-aware dependency installation
- NumPy 1.26+ is now installed with the core package instead of requiring an extra.
- `pyaccelerate info` checks core, GPU, and NPU dependencies and offers every missing package supported by the detected hardware.
- `pyaccelerate info --install-deps` installs all recommendations into the active Python interpreter without an interactive prompt.
- Non-interactive `info` runs report the exact repair command instead of silently skipping dependency guidance.

### Comprehensive GPU Hardware Detection
- **NVIDIA Architecture Database**: Full classification from Kepler (3.0) through Blackwell (10.2)
- **AMD Architecture Database**: 31 name patterns covering GCN 4 to RDNA 4 and CDNA 1-3
- **CUDA Core Counts**: Accurate per-SM core calculation for all NVIDIA architectures
- **Tensor Cores**: Detected on RTX and data-center GPUs; correctly absent on GTX 16xx
- **RT Cores**: Detected on RTX cards; correctly absent on GTX and data-center GPUs
- **NVENC/NVDEC**: Hardware video encode/decode detection (Maxwell+)
- **AMD VCN**: Video Core Next encode/decode detection (RDNA+)
- **Memory Type**: GDDR5/6/6X, HBM2/2e/3/3e classification per architecture
- **Memory Bandwidth**: Calculated from bus width x memory clock
- **Clock Speeds**: Base and boost clocks from CUDA and nvidia-smi
- **PCIe Info**: Generation and link width from nvidia-smi
- **Driver Version**: GPU driver and CUDA runtime version
- **Power Limit**: TDP from nvidia-smi
- **Copy Engines**: Async DMA engine count from CUDA
- **Shared VRAM for ALL vendors**: Now detects shared system memory for NVIDIA and AMD on Windows (previously Intel-only)
- **Features Property**: Capability flags list (compute, tensor, hw_encode, cuda_7.5, turing, etc.)

### Integration
- **Engine**: Enhanced summary with architecture, cores, features, clocks, PCIe, driver info
- **CLI gpu command**: Completely rewritten with full hardware report
- **Autotune**: TuneProfile expanded with 16 new GPU fields for persistent hardware profiling
- **Memory Stats**: get_gpu_memory_stats() now exports CUDA cores, tensor/RT cores, bandwidth, features

### Tests
- 33 GPU-specific tests (NVIDIA classification x 6, AMD classification x 6, GPUDevice x 21)
- 468/470 total tests passing (2 pre-existing flaky network timeouts)

### Example Output (GTX 1660 SUPER)
```
[0] NVIDIA GeForce GTX 1660 SUPER (CUDA, 6.0 GB +28.0 GB shared Turing Vulkan)
    Arch: Turing | Compute Capability 7.5
    Cores: CUDA=1408 (SMs=22)
    VRAM: 6.0 GB + 28.0 GB shared = 34.0 GB total
    Memory: GDDR6 | 192-bit bus | 336 GB/s bandwidth
    Clock: Base: 1815 MHz | Boost: 2130 MHz
    Features: HW Encode (NVENC), HW Decode (NVDEC), Copy Engines x2
    Hardware: PCIe Gen3 x16 | TDP 125W
    Driver: 595.79 | CUDA: 13.2
    Vulkan: 1.4.329
```
