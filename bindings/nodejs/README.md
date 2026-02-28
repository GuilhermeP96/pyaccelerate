# PyAccelerate — Node.js Client

Node.js/TypeScript client for the [PyAccelerate](https://github.com/GuilhermeP96/pyaccelerate) HTTP API server.

## Prerequisites

Start the PyAccelerate server first:

```bash
pip install pyaccelerate
pyaccelerate serve --http-port 8420
```

## Installation

```bash
npm install pyaccelerate
```

## Quick Start

```javascript
const { PyAccelerate } = require('pyaccelerate');

const client = new PyAccelerate('http://localhost:8420');

// Get full hardware info
const info = await client.getInfo();
console.log(`CPU: ${info.cpu.brand} (${info.cpu.logical_cores} threads)`);
console.log(`GPU: ${info.gpu.best?.name || 'N/A'}`);

// Get Prometheus metrics
const metrics = await client.getMetrics();

// Run benchmarks
const bench = await client.runBenchmark();
console.log(`CPU score: ${bench.cpu_single.math_ops_per_sec} ops/s`);

// Auto-tune
const profile = await client.runTune();
console.log(`Overall score: ${profile.overall_score}/100`);
```

## TypeScript

Full TypeScript declarations are included:

```typescript
import { PyAccelerate, CPUInfo, GPUDevice } from 'pyaccelerate';

const client = new PyAccelerate();
const cpu: CPUInfo = await client.getCPU();
```

## API

| Method | Description |
|---|---|
| `getInfo()` | Full hardware info (machine-readable) |
| `getSummary()` | Human-readable hardware summary |
| `getCPU()` | CPU details |
| `getGPU()` | GPU details |
| `getNPU()` | NPU details |
| `getMemory()` | Memory stats |
| `getVirt()` | Virtualization info |
| `getStatus()` | One-line status |
| `getMetrics()` | Prometheus metrics (text) |
| `runBenchmark(full?)` | Run micro-benchmarks |
| `runTune()` | Run auto-tune |
| `getTuneProfile()` | Get current tune profile |
| `health()` | Health check |
| `listEndpoints()` | List available endpoints |

## License

MIT
