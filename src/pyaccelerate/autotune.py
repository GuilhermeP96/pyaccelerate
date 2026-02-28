"""
pyaccelerate.autotune — Auto-tuning feedback loop.

Runs benchmarks, persists a *TuneProfile* with the optimal configuration,
and re-applies it on subsequent runs.  When the hardware fingerprint
changes (e.g. new GPU, RAM upgrade) the profile is automatically
invalidated and a new tuning cycle is triggered.

Usage::

    from pyaccelerate.autotune import auto_tune, get_or_tune, apply_profile

    # Run a full tune cycle (benchmark → save → return profile)
    profile = auto_tune()

    # Load existing profile or tune if stale / missing
    profile = get_or_tune()

    # Apply the profile to the current process
    result = apply_profile()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("pyaccelerate.autotune")

# ── Persistent storage ────────────────────────────────────────────────────

TUNE_DIR: Path = Path.home() / ".pyaccelerate"
TUNE_FILE: Path = TUNE_DIR / "tune_profile.json"

# Profile is considered stale after this many hours
DEFAULT_STALE_HOURS: int = 168  # 7 days


# ── Data model ────────────────────────────────────────────────────────────

@dataclass
class TuneProfile:
    """Persisted auto-tuning results."""

    # Metadata
    timestamp: str = ""
    hardware_hash: str = ""
    pyaccelerate_version: str = ""
    tune_duration_s: float = 0.0

    # Raw scores
    cpu_single_ops_sec: int = 0
    cpu_multi_ops_sec: int = 0
    memory_write_gbps: float = 0.0
    memory_read_gbps: float = 0.0
    thread_latency_avg_us: float = 0.0
    thread_latency_p95_us: float = 0.0
    gpu_available: bool = False
    gpu_gflops: float = 0.0
    gpu_backend: str = ""

    # Derived recommendations
    optimal_io_workers: int = 0
    optimal_cpu_workers: int = 0
    optimal_gpu_strategy: str = "round-robin"
    recommended_priority: str = "NORMAL"
    recommended_energy: str = "BALANCED"
    memory_pressure_headroom_gb: float = 0.0

    # Scores 0 – 100  (composite)
    cpu_score: int = 0
    gpu_score: int = 0
    memory_score: int = 0
    overall_score: int = 0


# ── Hardware fingerprinting ───────────────────────────────────────────────

def hardware_fingerprint() -> str:
    """Return a short hash summarising the current hardware configuration.

    Changes when CPU model, core count, total RAM, or GPU list changes.
    """
    parts: list[str] = []

    try:
        from pyaccelerate.cpu import detect as _cpu
        c = _cpu()
        parts.append(f"cpu:{c.brand}:{c.physical_cores}:{c.logical_cores}")
    except Exception:
        parts.append("cpu:unknown")

    try:
        import psutil  # type: ignore[import-untyped]
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        parts.append(f"ram:{ram_gb}")
    except Exception:
        parts.append("ram:unknown")

    try:
        from pyaccelerate.gpu import detect_all as _gpus
        gpu_names = sorted(g.name for g in _gpus())
        parts.append(f"gpu:{','.join(gpu_names) or 'none'}")
    except Exception:
        parts.append("gpu:unknown")

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Scoring helpers ───────────────────────────────────────────────────────

def _cpu_score(single_ops: int, multi_ops: int) -> int:
    """0-100 score based on ops/sec.  Calibrated to i9-13900K ≈ 95."""
    # single: ~2.5M baseline, ~8M top
    s = min(100, int(single_ops / 80_000))
    # multi: ~15M baseline, ~80M top
    m = min(100, int(multi_ops / 800_000))
    return min(100, (s + m) // 2)


def _memory_score(write_gbps: float, read_gbps: float) -> int:
    """0-100 score based on bandwidth.  ~50 GB/s DDR5 ≈ 90."""
    avg = (write_gbps + read_gbps) / 2
    return min(100, int(avg / 0.5))  # 50 GB/s → 100


def _gpu_score(gflops: float) -> int:
    """0-100 score.  RTX 4090 ~80 TFLOPS (80k GFLOPS) ≈ 100."""
    if gflops <= 0:
        return 0
    return min(100, max(1, int(gflops / 800)))


def _derive_workers(cpu_cores: int, mem_available_gb: float, latency_us: float) -> tuple[int, int]:
    """Determine optimal IO & CPU workers based on benchmarks."""
    # IO workers: scale with cores, clamp by memory
    io_base = min(cpu_cores * 3, 64)
    if mem_available_gb < 2:
        io_base = min(io_base, 4)
    elif mem_available_gb < 4:
        io_base = min(io_base, 8)
    elif mem_available_gb < 8:
        io_base = min(io_base, 16)

    # If latency is high, the pool isn't handling well — reduce
    if latency_us > 500:
        io_base = max(2, io_base // 2)

    # CPU workers: physical cores (not logical) for compute
    cpu_w = max(1, cpu_cores)

    return io_base, cpu_w


def _derive_priority(overall: int) -> tuple[str, str]:
    """Suggest priority/energy based on overall score."""
    if overall >= 80:
        return "NORMAL", "BALANCED"
    elif overall >= 50:
        return "ABOVE_NORMAL", "PERFORMANCE"
    else:
        return "HIGH", "ULTRA_PERFORMANCE"


# ── Core functions ────────────────────────────────────────────────────────

def auto_tune(*, quick: bool = True) -> TuneProfile:
    """Run benchmarks and return an optimal :class:`TuneProfile`.

    The profile is automatically saved to ``~/.pyaccelerate/tune_profile.json``.

    Parameters
    ----------
    quick : bool
        If *True* (default), use reduced iteration counts (~5 s).
        Set to *False* for a thorough benchmark run.
    """
    from pyaccelerate import __version__
    from pyaccelerate.benchmark import run_all

    log.info("Starting auto-tune (quick=%s) …", quick)
    t0 = time.perf_counter()

    results = run_all(quick=quick)
    elapsed = time.perf_counter() - t0

    # Extract raw numbers
    cpu_s = results.get("cpu_single", {})
    cpu_m = results.get("cpu_multi", {})
    mem = results.get("memory", {})
    thr = results.get("thread_latency", {})
    gpu = results.get("gpu", {})

    single_ops = cpu_s.get("math_ops_per_sec", 0)
    multi_ops = cpu_m.get("ops_per_sec", 0)
    write_gbps = mem.get("write_gbps", 0.0)
    read_gbps = mem.get("read_gbps", 0.0)
    lat_avg = thr.get("avg_latency_us", 0.0)
    lat_p95 = thr.get("p95_latency_us", 0.0)
    gpu_ok = gpu.get("available", False)
    gpu_gf = gpu.get("gflops", 0.0)
    gpu_be = gpu.get("backend", "")

    # Scores
    cs = _cpu_score(single_ops, multi_ops)
    ms = _memory_score(write_gbps, read_gbps)
    gs = _gpu_score(gpu_gf)
    overall = (cs * 4 + ms * 2 + gs * 4) // 10 if gpu_ok else (cs * 6 + ms * 4) // 10

    # Workers
    try:
        from pyaccelerate.cpu import detect as _cpu
        c = _cpu()
        phys = c.physical_cores
    except Exception:
        phys = os.cpu_count() or 4

    try:
        import psutil  # type: ignore[import-untyped]
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        avail_gb = 8.0

    io_w, cpu_w = _derive_workers(phys, avail_gb, lat_avg)
    prio, energy = _derive_priority(overall)

    profile = TuneProfile(
        timestamp=datetime.now(timezone.utc).isoformat(),
        hardware_hash=hardware_fingerprint(),
        pyaccelerate_version=__version__,
        tune_duration_s=round(elapsed, 2),
        cpu_single_ops_sec=single_ops,
        cpu_multi_ops_sec=multi_ops,
        memory_write_gbps=write_gbps,
        memory_read_gbps=read_gbps,
        thread_latency_avg_us=lat_avg,
        thread_latency_p95_us=lat_p95,
        gpu_available=gpu_ok,
        gpu_gflops=gpu_gf,
        gpu_backend=gpu_be,
        optimal_io_workers=io_w,
        optimal_cpu_workers=cpu_w,
        optimal_gpu_strategy="score-weighted" if gpu_ok else "round-robin",
        recommended_priority=prio,
        recommended_energy=energy,
        memory_pressure_headroom_gb=round(avail_gb, 2),
        cpu_score=cs,
        gpu_score=gs,
        memory_score=ms,
        overall_score=overall,
    )

    save_profile(profile)
    log.info(
        "Auto-tune complete in %.1f s — overall score %d/100 (CPU %d, MEM %d, GPU %d)",
        elapsed, overall, cs, ms, gs,
    )
    return profile


def save_profile(profile: TuneProfile) -> None:
    """Persist a :class:`TuneProfile` to disk."""
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    data = asdict(profile)
    TUNE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.debug("Saved tune profile → %s", TUNE_FILE)


def load_profile() -> Optional[TuneProfile]:
    """Load a previously saved :class:`TuneProfile`, or *None*."""
    if not TUNE_FILE.exists():
        return None
    try:
        data = json.loads(TUNE_FILE.read_text(encoding="utf-8"))
        return TuneProfile(**{k: v for k, v in data.items() if k in TuneProfile.__dataclass_fields__})
    except Exception as exc:
        log.warning("Failed to load tune profile: %s", exc)
        return None


def needs_retune(*, stale_hours: int = DEFAULT_STALE_HOURS) -> bool:
    """Check whether a retune is recommended.

    Returns *True* if:
    - No profile exists
    - Hardware fingerprint changed
    - Profile is older than *stale_hours*
    """
    profile = load_profile()
    if profile is None:
        return True

    # Hardware changed?
    if profile.hardware_hash != hardware_fingerprint():
        log.info("Hardware fingerprint changed — retune recommended")
        return True

    # Stale?
    try:
        ts = datetime.fromisoformat(profile.timestamp)
        age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        if age_hours > stale_hours:
            log.info("Profile is %.0f h old (limit %d h) — retune recommended", age_hours, stale_hours)
            return True
    except Exception:
        return True

    return False


def get_or_tune(*, quick: bool = True, stale_hours: int = DEFAULT_STALE_HOURS) -> TuneProfile:
    """Return a valid :class:`TuneProfile`, tuning first if necessary."""
    if not needs_retune(stale_hours=stale_hours):
        profile = load_profile()
        if profile is not None:
            return profile
    return auto_tune(quick=quick)


def apply_profile(profile: Optional[TuneProfile] = None) -> Dict[str, Any]:
    """Apply a :class:`TuneProfile` to the running process.

    Sets thread pool sizes, OS priority, and energy profile.
    Returns a dict of what was applied.

    If *profile* is ``None``, loads from disk (or tunes).
    """
    if profile is None:
        profile = get_or_tune()

    applied: Dict[str, Any] = {}

    # Thread pool environment hints (picked up by threads.py on next get_pool)
    if profile.optimal_io_workers > 0:
        os.environ["PYACCELERATE_IO_WORKERS"] = str(profile.optimal_io_workers)
        applied["io_workers"] = profile.optimal_io_workers
    if profile.optimal_cpu_workers > 0:
        os.environ["PYACCELERATE_CPU_WORKERS"] = str(profile.optimal_cpu_workers)
        applied["cpu_workers"] = profile.optimal_cpu_workers

    # Priority
    try:
        from pyaccelerate.priority import TaskPriority, set_task_priority
        tp = TaskPriority[profile.recommended_priority]
        ok = set_task_priority(tp)
        applied["priority"] = {"level": profile.recommended_priority, "applied": ok}
    except Exception as exc:
        applied["priority"] = {"error": str(exc)}

    # Energy
    try:
        from pyaccelerate.priority import EnergyProfile, set_energy_profile
        ep = EnergyProfile[profile.recommended_energy]
        ok = set_energy_profile(ep)
        applied["energy"] = {"profile": profile.recommended_energy, "applied": ok}
    except Exception as exc:
        applied["energy"] = {"error": str(exc)}

    applied["overall_score"] = profile.overall_score
    log.info("Profile applied — score %d/100", profile.overall_score)
    return applied


def delete_profile() -> bool:
    """Delete the persisted tune profile (forces retune on next call)."""
    if TUNE_FILE.exists():
        TUNE_FILE.unlink()
        log.info("Deleted tune profile")
        return True
    return False


def profile_summary(profile: Optional[TuneProfile] = None) -> str:
    """Human-readable summary of a tune profile."""
    if profile is None:
        profile = load_profile()
    if profile is None:
        return "No tune profile found. Run `pyaccelerate tune` to create one."

    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║              PyAccelerate — Tune Profile                    ║",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  Timestamp:  {profile.timestamp}",
        f"║  Duration:   {profile.tune_duration_s:.1f} s",
        f"║  HW hash:    {profile.hardware_hash}",
        f"║  Version:    {profile.pyaccelerate_version}",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  Overall score:  {profile.overall_score}/100",
        f"║  CPU score:      {profile.cpu_score}/100  "
        f"(single {profile.cpu_single_ops_sec:,} ops/s | multi {profile.cpu_multi_ops_sec:,} ops/s)",
        f"║  Memory score:   {profile.memory_score}/100  "
        f"(W {profile.memory_write_gbps:.1f} GB/s | R {profile.memory_read_gbps:.1f} GB/s)",
        f"║  GPU score:      {profile.gpu_score}/100  "
        f"({profile.gpu_gflops:.1f} GFLOPS — {profile.gpu_backend or 'N/A'})",
        "╠══════════════════════════════════════════════════════════════╣",
        f"║  IO workers:     {profile.optimal_io_workers}",
        f"║  CPU workers:    {profile.optimal_cpu_workers}",
        f"║  GPU strategy:   {profile.optimal_gpu_strategy}",
        f"║  Priority:       {profile.recommended_priority}",
        f"║  Energy:         {profile.recommended_energy}",
        f"║  Thread P95:     {profile.thread_latency_p95_us:.0f} µs",
        f"║  RAM headroom:   {profile.memory_pressure_headroom_gb:.1f} GB",
        "╚══════════════════════════════════════════════════════════════╝",
    ]
    return "\n".join(lines)
