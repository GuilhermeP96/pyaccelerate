"""
pyaccelerate.k8s — Kubernetes integration for GPU workload auto-scaling.

Detects when running inside a Kubernetes pod, reads resource requests /
limits, probes node GPU capacity, and generates scaling recommendations.

Requires the ``kubernetes`` client library for full API access, but basic
environment detection works with stdlib alone.

Usage::

    from pyaccelerate.k8s import (
        is_kubernetes,
        get_pod_info,
        get_node_gpu_capacity,
        get_scaling_recommendation,
        generate_resource_manifest,
    )

    if is_kubernetes():
        pod = get_pod_info()
        rec = get_scaling_recommendation()
        manifest = generate_resource_manifest()

CLI::

    pyaccelerate k8s          # Show Kubernetes pod & GPU info
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("pyaccelerate.k8s")


# ── Data models ───────────────────────────────────────────────────────────

@dataclass
class PodInfo:
    """Information about the current Kubernetes pod."""

    name: str = ""
    namespace: str = ""
    node_name: str = ""
    service_account: str = ""
    cpu_request: str = ""
    cpu_limit: str = ""
    memory_request: str = ""
    memory_limit: str = ""
    gpu_request: int = 0
    gpu_limit: int = 0
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class NodeGPU:
    """GPU capacity on a Kubernetes node."""

    node_name: str = ""
    gpu_type: str = ""
    gpu_product: str = ""
    total: int = 0
    allocatable: int = 0
    allocated: int = 0
    available: int = 0


@dataclass
class ScalingRecommendation:
    """Auto-scaling recommendation based on hardware probing."""

    current_replicas: int = 1
    recommended_replicas: int = 1
    gpu_per_replica: int = 0
    cpu_per_replica: str = ""
    memory_per_replica: str = ""
    reason: str = ""
    gpu_utilization_pct: float = 0.0
    scale_direction: str = "none"  # "up", "down", "none"


# ── Environment detection ─────────────────────────────────────────────────

def is_kubernetes() -> bool:
    """Return *True* if running inside a Kubernetes pod."""
    # K8s sets this env var in every pod
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    # Also check for the service account token
    return Path("/var/run/secrets/kubernetes.io/serviceaccount/token").exists()


def get_namespace() -> str:
    """Return the current Kubernetes namespace (or ``'default'``)."""
    ns_file = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
    if ns_file.exists():
        return ns_file.read_text().strip()
    return os.environ.get("POD_NAMESPACE", "default")


def get_pod_name() -> str:
    """Return the current pod name (from ``$HOSTNAME`` or downward API)."""
    return os.environ.get("POD_NAME", os.environ.get("HOSTNAME", "unknown"))


# ── Pod info ──────────────────────────────────────────────────────────────

def get_pod_info() -> PodInfo:
    """Gather information about the current pod.

    Uses the Kubernetes downward API environment variables first, then
    falls back to the ``kubernetes`` client library.
    """
    info = PodInfo(
        name=get_pod_name(),
        namespace=get_namespace(),
        node_name=os.environ.get("NODE_NAME", ""),
        service_account=os.environ.get("SERVICE_ACCOUNT", ""),
    )

    # Resource limits from downward API or cgroup
    info.cpu_request = os.environ.get("CPU_REQUEST", "")
    info.cpu_limit = os.environ.get("CPU_LIMIT", _read_cpu_limit_cgroup())
    info.memory_request = os.environ.get("MEMORY_REQUEST", "")
    info.memory_limit = os.environ.get("MEMORY_LIMIT", _read_memory_limit_cgroup())

    # GPU from env (NVIDIA device plugin)
    nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    if nvidia_visible and nvidia_visible != "none":
        gpus = [d.strip() for d in nvidia_visible.split(",") if d.strip()]
        info.gpu_limit = len(gpus)
        info.gpu_request = len(gpus)

    # Try kubernetes client for richer data
    try:
        info = _enrich_from_k8s_api(info)
    except Exception:
        pass

    return info


def _read_cpu_limit_cgroup() -> str:
    """Read CPU limit from cgroup (v2 first, then v1)."""
    # cgroup v2
    cg2 = Path("/sys/fs/cgroup/cpu.max")
    if cg2.exists():
        try:
            parts = cg2.read_text().strip().split()
            if parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1]) if len(parts) > 1 else 100000
                return f"{quota / period:.1f}"
        except Exception:
            pass

    # cgroup v1
    quota_file = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_file = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_file.exists() and period_file.exists():
        try:
            quota = int(quota_file.read_text().strip())
            period = int(period_file.read_text().strip())
            if quota > 0:
                return f"{quota / period:.1f}"
        except Exception:
            pass

    return ""


def _read_memory_limit_cgroup() -> str:
    """Read memory limit from cgroup."""
    # cgroup v2
    cg2 = Path("/sys/fs/cgroup/memory.max")
    if cg2.exists():
        try:
            val = cg2.read_text().strip()
            if val != "max":
                mb = int(val) / (1024 ** 2)
                return f"{mb:.0f}Mi"
        except Exception:
            pass

    # cgroup v1
    limit_file = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit_file.exists():
        try:
            val = int(limit_file.read_text().strip())
            # Linux returns a huge value if no limit
            if val < 2 ** 62:
                mb = val / (1024 ** 2)
                return f"{mb:.0f}Mi"
        except Exception:
            pass

    return ""


def _enrich_from_k8s_api(info: PodInfo) -> PodInfo:
    """Enrich pod info using the kubernetes Python client."""
    try:
        from kubernetes import client, config  # type: ignore[import-untyped]
    except ImportError:
        return info

    try:
        config.load_incluster_config()
    except Exception:
        try:
            config.load_kube_config()
        except Exception:
            return info

    try:
        v1 = client.CoreV1Api()
        pod = v1.read_namespaced_pod(info.name, info.namespace)

        info.node_name = pod.spec.node_name or info.node_name
        info.service_account = pod.spec.service_account_name or info.service_account
        info.labels = dict(pod.metadata.labels or {})
        info.annotations = dict(pod.metadata.annotations or {})

        # Parse container resources
        for container in pod.spec.containers:
            res = container.resources
            if res:
                req = res.requests or {}
                lim = res.limits or {}
                if not info.cpu_request and "cpu" in req:
                    info.cpu_request = str(req["cpu"])
                if not info.cpu_limit and "cpu" in lim:
                    info.cpu_limit = str(lim["cpu"])
                if not info.memory_request and "memory" in req:
                    info.memory_request = str(req["memory"])
                if not info.memory_limit and "memory" in lim:
                    info.memory_limit = str(lim["memory"])
                gpu_key = "nvidia.com/gpu"
                if not info.gpu_request and gpu_key in req:
                    info.gpu_request = int(req[gpu_key])
                if not info.gpu_limit and gpu_key in lim:
                    info.gpu_limit = int(lim[gpu_key])
    except Exception as exc:
        log.debug("K8s API enrichment failed: %s", exc)

    return info


# ── Node GPU capacity ─────────────────────────────────────────────────────

def get_node_gpu_capacity() -> List[NodeGPU]:
    """Query GPU capacity across cluster nodes.

    Requires the ``kubernetes`` client and appropriate RBAC.
    """
    nodes: List[NodeGPU] = []

    try:
        from kubernetes import client, config  # type: ignore[import-untyped]
    except ImportError:
        log.warning("kubernetes client not installed — cannot query node GPUs")
        return nodes

    try:
        config.load_incluster_config()
    except Exception:
        try:
            config.load_kube_config()
        except Exception:
            return nodes

    try:
        v1 = client.CoreV1Api()
        node_list = v1.list_node()

        for node in node_list.items:
            node_name = node.metadata.name
            cap = node.status.capacity or {}
            alloc = node.status.allocatable or {}

            gpu_cap = int(cap.get("nvidia.com/gpu", 0))
            gpu_alloc = int(alloc.get("nvidia.com/gpu", 0))

            if gpu_cap > 0:
                # Try to get GPU product from labels
                labels = node.metadata.labels or {}
                gpu_product = labels.get(
                    "nvidia.com/gpu.product",
                    labels.get("accelerator", "unknown"),
                )
                gpu_type = labels.get("nvidia.com/gpu.memory", "")

                nodes.append(NodeGPU(
                    node_name=node_name,
                    gpu_type=gpu_type,
                    gpu_product=gpu_product,
                    total=gpu_cap,
                    allocatable=gpu_alloc,
                    allocated=gpu_cap - gpu_alloc,
                    available=gpu_alloc,
                ))
    except Exception as exc:
        log.warning("Failed to query node GPUs: %s", exc)

    return nodes


# ── Scaling recommendation ────────────────────────────────────────────────

def get_scaling_recommendation() -> ScalingRecommendation:
    """Generate an auto-scaling recommendation based on current hardware.

    Considers GPU availability, memory pressure, and CPU utilisation.
    """
    rec = ScalingRecommendation()

    # Detect local hardware
    try:
        from pyaccelerate.gpu import detect_all
        gpus = detect_all()
        gpu_count = len([g for g in gpus if g.usable])
    except Exception:
        gpu_count = 0

    try:
        from pyaccelerate.cpu import detect as detect_cpu
        cpu = detect_cpu()
        cores = cpu.physical_cores
    except Exception:
        cores = os.cpu_count() or 1

    try:
        import psutil  # type: ignore[import-untyped]
        mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_pct = psutil.cpu_percent(interval=0.5)
    except Exception:
        mem_gb = 8.0
        cpu_pct = 50.0

    rec.gpu_utilization_pct = cpu_pct  # approximate

    # GPU-based recommendation
    if gpu_count > 0:
        rec.gpu_per_replica = 1
        rec.recommended_replicas = gpu_count
        rec.reason = f"{gpu_count} GPU(s) available — 1 replica per GPU"
        if gpu_count > 1:
            rec.scale_direction = "up"
    else:
        # CPU-based: 1 replica per 4 cores, minimum 1
        rec.recommended_replicas = max(1, cores // 4)
        rec.cpu_per_replica = f"{min(4, cores)}"
        rec.reason = f"{cores} CPU cores — {rec.recommended_replicas} replica(s) recommended"

    # Memory: at least 2 GB per replica
    max_by_mem = max(1, int(mem_gb / 2))
    if rec.recommended_replicas > max_by_mem:
        rec.recommended_replicas = max_by_mem
        rec.reason += f" (capped by {mem_gb:.0f} GB RAM)"

    rec.memory_per_replica = f"{max(512, int(mem_gb * 1024 / rec.recommended_replicas))}Mi"

    if rec.recommended_replicas > rec.current_replicas:
        rec.scale_direction = "up"
    elif rec.recommended_replicas < rec.current_replicas:
        rec.scale_direction = "down"

    return rec


# ── Manifest generation ───────────────────────────────────────────────────

def generate_resource_manifest(
    name: str = "pyaccelerate-worker",
    image: str = "pyaccelerate:latest",
    namespace: str = "default",
    replicas: int = 0,
    gpu_per_replica: int = 0,
) -> str:
    """Generate a Kubernetes Deployment YAML manifest.

    If *replicas* or *gpu_per_replica* are 0, uses values from
    :func:`get_scaling_recommendation`.
    """
    rec = get_scaling_recommendation()
    if replicas <= 0:
        replicas = rec.recommended_replicas
    if gpu_per_replica <= 0:
        gpu_per_replica = rec.gpu_per_replica

    gpu_resource = ""
    if gpu_per_replica > 0:
        gpu_resource = f"""
            nvidia.com/gpu: "{gpu_per_replica}"
"""

    manifest = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  namespace: {namespace}
  labels:
    app: {name}
    managed-by: pyaccelerate
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
      - name: worker
        image: {image}
        resources:
          requests:
            cpu: "{rec.cpu_per_replica or '1'}"
            memory: "{rec.memory_per_replica or '512Mi'}"
          limits:
            cpu: "{rec.cpu_per_replica or '2'}"
            memory: "{rec.memory_per_replica or '1Gi'}"{gpu_resource}
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
"""

    # Add GPU tolerations/nodeSelector if GPUs requested
    if gpu_per_replica > 0:
        manifest += """      nodeSelector:
        nvidia.com/gpu.present: "true"
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
"""

    return manifest


# ── Summary ───────────────────────────────────────────────────────────────

def get_k8s_summary() -> Dict[str, Any]:
    """Return a summary dict of Kubernetes environment."""
    result: Dict[str, Any] = {"is_kubernetes": is_kubernetes()}

    if not is_kubernetes():
        return result

    pod = get_pod_info()
    result["pod"] = asdict(pod)

    try:
        gpu_nodes = get_node_gpu_capacity()
        result["gpu_nodes"] = [asdict(n) for n in gpu_nodes]
    except Exception:
        result["gpu_nodes"] = []

    rec = get_scaling_recommendation()
    result["scaling"] = asdict(rec)

    return result
