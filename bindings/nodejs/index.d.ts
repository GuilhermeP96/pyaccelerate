/**
 * TypeScript declarations for the PyAccelerate Node.js client.
 */

export interface PyAccelerateOptions {
  timeout?: number;
  headers?: Record<string, string>;
}

export interface CPUInfo {
  brand: string;
  arch: string;
  physical_cores: number;
  logical_cores: number;
  frequency_mhz: number;
  frequency_max_mhz: number;
  numa_nodes: number;
  smt_ratio: number;
  is_arm: boolean;
  is_android: boolean;
  is_sbc: boolean;
  flags: string[];
}

export interface GPUDevice {
  name: string;
  vendor: string;
  backend: string;
  memory_gb: number;
  compute_units: number;
  score: number;
  usable: boolean;
  is_discrete: boolean;
}

export interface NPUDevice {
  name: string;
  vendor: string;
  backend: string;
  tops: number;
  score: number;
  usable: boolean;
  driver_version: string | null;
}

export interface MemoryStats {
  stats: Record<string, number>;
  pressure: string;
}

export interface VirtInfo {
  vtx: boolean;
  hyperv: boolean;
  kvm: boolean;
  wsl: boolean;
  docker: boolean;
  inside_container: boolean;
  summary: string[];
}

export interface EngineInfo {
  cpu: CPUInfo;
  gpu: {
    enabled: boolean;
    multi_gpu: boolean;
    devices: GPUDevice[];
    best: GPUDevice | null;
  };
  npu: {
    enabled: boolean;
    devices: NPUDevice[];
    best: NPUDevice | null;
  };
  memory: Record<string, number>;
  memory_pressure: string;
  virt: VirtInfo;
  pools: {
    io_workers: number;
    cpu_workers: number;
  };
}

export interface BenchmarkResult {
  cpu_single: Record<string, any>;
  cpu_multi: Record<string, any>;
  thread_latency: Record<string, any>;
  memory: Record<string, any>;
  gpu: Record<string, any>;
}

export interface TuneProfile {
  profiled: boolean;
  timestamp?: string;
  hardware_hash?: string;
  overall_score?: number;
  cpu_score?: number;
  memory_score?: number;
  gpu_score?: number;
  optimal_io_workers?: number;
  optimal_cpu_workers?: number;
  recommended_priority?: string;
  recommended_energy?: string;
}

export interface EndpointList {
  endpoints: {
    GET: string[];
    POST: string[];
  };
}

export declare class PyAccelerateError extends Error {
  statusCode?: number;
  body?: any;
}

export declare class PyAccelerate {
  constructor(baseUrl?: string, options?: PyAccelerateOptions);

  baseUrl: string;
  timeout: number;
  headers: Record<string, string>;

  getInfo(): Promise<EngineInfo>;
  getSummary(): Promise<{ summary: string }>;
  getCPU(): Promise<CPUInfo>;
  getGPU(): Promise<{ count: number; devices: GPUDevice[] }>;
  getNPU(): Promise<{ count: number; devices: NPUDevice[] }>;
  getMemory(): Promise<MemoryStats>;
  getVirt(): Promise<VirtInfo>;
  getStatus(): Promise<{ status: string }>;
  getMetrics(): Promise<string>;
  runBenchmark(full?: boolean): Promise<BenchmarkResult>;
  runTune(): Promise<TuneProfile>;
  getTuneProfile(): Promise<TuneProfile>;
  health(): Promise<string>;
  listEndpoints(): Promise<EndpointList>;
}
