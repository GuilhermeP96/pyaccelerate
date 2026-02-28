"""Tests for pyaccelerate.autotune — Auto-tuning feedback loop."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pyaccelerate.autotune import (
    TuneProfile,
    hardware_fingerprint,
    auto_tune,
    save_profile,
    load_profile,
    needs_retune,
    get_or_tune,
    apply_profile,
    delete_profile,
    profile_summary,
    _cpu_score,
    _memory_score,
    _gpu_score,
    _derive_workers,
    _derive_priority,
    TUNE_DIR,
    TUNE_FILE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _tmp_tune_dir(tmp_path, monkeypatch):
    """Redirect tune storage to a temp directory."""
    tune_dir = tmp_path / ".pyaccelerate"
    tune_file = tune_dir / "tune_profile.json"
    monkeypatch.setattr("pyaccelerate.autotune.TUNE_DIR", tune_dir)
    monkeypatch.setattr("pyaccelerate.autotune.TUNE_FILE", tune_file)
    return tune_dir


# ── TuneProfile ───────────────────────────────────────────────────────────

class TestTuneProfile:
    def test_defaults(self):
        p = TuneProfile()
        assert p.overall_score == 0
        assert p.timestamp == ""
        assert p.gpu_available is False

    def test_asdict(self):
        p = TuneProfile(cpu_score=50, memory_score=60, gpu_score=70)
        d = asdict(p)
        assert d["cpu_score"] == 50
        assert d["memory_score"] == 60
        assert d["gpu_score"] == 70

    def test_all_fields_present(self):
        p = TuneProfile()
        d = asdict(p)
        expected_keys = {
            "timestamp", "hardware_hash", "pyaccelerate_version", "tune_duration_s",
            "cpu_single_ops_sec", "cpu_multi_ops_sec", "memory_write_gbps",
            "memory_read_gbps", "thread_latency_avg_us", "thread_latency_p95_us",
            "gpu_available", "gpu_gflops", "gpu_backend", "optimal_io_workers",
            "optimal_cpu_workers", "optimal_gpu_strategy", "recommended_priority",
            "recommended_energy", "memory_pressure_headroom_gb",
            "cpu_score", "gpu_score", "memory_score", "overall_score",
        }
        assert set(d.keys()) == expected_keys


# ── Scoring ───────────────────────────────────────────────────────────────

class TestScoring:
    def test_cpu_score_zero(self):
        assert _cpu_score(0, 0) == 0

    def test_cpu_score_moderate(self):
        score = _cpu_score(2_000_000, 10_000_000)
        assert 10 <= score <= 80

    def test_cpu_score_capped(self):
        score = _cpu_score(100_000_000, 500_000_000)
        assert score == 100

    def test_memory_score_moderate(self):
        score = _memory_score(10.0, 15.0)
        assert 10 <= score <= 100

    def test_memory_score_zero(self):
        assert _memory_score(0, 0) == 0

    def test_gpu_score_zero(self):
        assert _gpu_score(0) == 0

    def test_gpu_score_moderate(self):
        score = _gpu_score(500.0)
        assert 0 < score < 100

    def test_gpu_score_high(self):
        score = _gpu_score(100_000.0)
        assert score == 100


class TestDeriveWorkers:
    def test_low_memory(self):
        io, cpu = _derive_workers(4, 1.5, 100)
        assert io <= 4
        assert cpu == 4

    def test_normal_memory(self):
        io, cpu = _derive_workers(8, 16.0, 50)
        assert io >= 8
        assert cpu == 8

    def test_high_latency_clamp(self):
        io, cpu = _derive_workers(8, 16.0, 600)
        io_normal, _ = _derive_workers(8, 16.0, 50)
        assert io < io_normal

    def test_single_core(self):
        io, cpu = _derive_workers(1, 8.0, 50)
        assert io >= 1
        assert cpu == 1


class TestDerivePriority:
    def test_high_score(self):
        prio, energy = _derive_priority(90)
        assert prio == "NORMAL"
        assert energy == "BALANCED"

    def test_medium_score(self):
        prio, energy = _derive_priority(60)
        assert prio == "ABOVE_NORMAL"
        assert energy == "PERFORMANCE"

    def test_low_score(self):
        prio, energy = _derive_priority(30)
        assert prio == "HIGH"
        assert energy == "ULTRA_PERFORMANCE"


# ── Hardware fingerprinting ───────────────────────────────────────────────

class TestHardwareFingerprint:
    def test_returns_string(self):
        fp = hardware_fingerprint()
        assert isinstance(fp, str)
        assert len(fp) == 16

    def test_stable(self):
        fp1 = hardware_fingerprint()
        fp2 = hardware_fingerprint()
        assert fp1 == fp2


# ── Save / Load ──────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_creates_file(self, _tmp_tune_dir):
        p = TuneProfile(cpu_score=42, timestamp="2026-01-01T00:00:00+00:00")
        save_profile(p)
        tune_file = _tmp_tune_dir / "tune_profile.json"
        assert tune_file.exists()

    def test_load_roundtrip(self, _tmp_tune_dir):
        p = TuneProfile(cpu_score=77, memory_score=55, overall_score=66)
        save_profile(p)
        loaded = load_profile()
        assert loaded is not None
        assert loaded.cpu_score == 77
        assert loaded.memory_score == 55
        assert loaded.overall_score == 66

    def test_load_missing(self):
        loaded = load_profile()
        assert loaded is None

    def test_load_corrupted(self, _tmp_tune_dir):
        tune_file = _tmp_tune_dir / "tune_profile.json"
        _tmp_tune_dir.mkdir(parents=True, exist_ok=True)
        tune_file.write_text("NOT JSON")
        loaded = load_profile()
        assert loaded is None

    def test_delete_profile(self, _tmp_tune_dir):
        save_profile(TuneProfile())
        assert delete_profile() is True
        assert load_profile() is None

    def test_delete_missing(self):
        assert delete_profile() is False


# ── Needs retune ──────────────────────────────────────────────────────────

class TestNeedsRetune:
    def test_no_profile(self):
        assert needs_retune() is True

    def test_fresh_profile(self, _tmp_tune_dir):
        p = TuneProfile(
            timestamp=datetime.now(timezone.utc).isoformat(),
            hardware_hash=hardware_fingerprint(),
        )
        save_profile(p)
        assert needs_retune() is False

    def test_stale_profile(self, _tmp_tune_dir):
        old = datetime.now(timezone.utc) - timedelta(hours=200)
        p = TuneProfile(
            timestamp=old.isoformat(),
            hardware_hash=hardware_fingerprint(),
        )
        save_profile(p)
        assert needs_retune(stale_hours=168) is True

    def test_hardware_changed(self, _tmp_tune_dir):
        p = TuneProfile(
            timestamp=datetime.now(timezone.utc).isoformat(),
            hardware_hash="different_hash_123",
        )
        save_profile(p)
        assert needs_retune() is True


# ── Auto-tune (mocked benchmarks) ────────────────────────────────────────

class TestAutoTune:
    @patch("pyaccelerate.autotune.hardware_fingerprint", return_value="mock_fp_12345678")
    @patch("pyaccelerate.benchmark.run_all")
    def test_auto_tune_returns_profile(self, mock_bench, mock_fp, _tmp_tune_dir):
        mock_bench.return_value = {
            "cpu_single": {"math_ops_per_sec": 2_000_000, "hash_ops_per_sec": 1_000_000},
            "cpu_multi": {"ops_per_sec": 8_000_000},
            "memory": {"write_gbps": 5.0, "read_gbps": 8.0},
            "thread_latency": {"avg_latency_us": 50.0, "p95_latency_us": 100.0},
            "gpu": {"available": False},
        }
        profile = auto_tune(quick=True)
        assert profile.hardware_hash == "mock_fp_12345678"
        assert profile.cpu_score > 0
        assert profile.overall_score > 0
        assert profile.gpu_available is False

    @patch("pyaccelerate.autotune.hardware_fingerprint", return_value="mock_fp_12345678")
    @patch("pyaccelerate.benchmark.run_all")
    def test_auto_tune_with_gpu(self, mock_bench, mock_fp, _tmp_tune_dir):
        mock_bench.return_value = {
            "cpu_single": {"math_ops_per_sec": 3_000_000},
            "cpu_multi": {"ops_per_sec": 15_000_000},
            "memory": {"write_gbps": 10.0, "read_gbps": 20.0},
            "thread_latency": {"avg_latency_us": 30.0, "p95_latency_us": 60.0},
            "gpu": {"available": True, "gflops": 1500.0, "backend": "cuda"},
        }
        profile = auto_tune(quick=True)
        assert profile.gpu_available is True
        assert profile.gpu_gflops == 1500.0
        assert profile.gpu_backend == "cuda"
        assert profile.optimal_gpu_strategy == "score-weighted"

    @patch("pyaccelerate.autotune.hardware_fingerprint", return_value="mock_fp_12345678")
    @patch("pyaccelerate.benchmark.run_all")
    def test_auto_tune_saves(self, mock_bench, mock_fp, _tmp_tune_dir):
        mock_bench.return_value = {
            "cpu_single": {"math_ops_per_sec": 1_000_000},
            "cpu_multi": {"ops_per_sec": 4_000_000},
            "memory": {"write_gbps": 3.0, "read_gbps": 5.0},
            "thread_latency": {"avg_latency_us": 80.0, "p95_latency_us": 150.0},
            "gpu": {"available": False},
        }
        auto_tune(quick=True)
        loaded = load_profile()
        assert loaded is not None
        assert loaded.hardware_hash == "mock_fp_12345678"


# ── Get or tune ───────────────────────────────────────────────────────────

class TestGetOrTune:
    @patch("pyaccelerate.autotune.hardware_fingerprint", return_value="mock_fp_12345678")
    @patch("pyaccelerate.benchmark.run_all")
    def test_tunes_when_missing(self, mock_bench, mock_fp, _tmp_tune_dir):
        mock_bench.return_value = {
            "cpu_single": {"math_ops_per_sec": 1_000_000},
            "cpu_multi": {"ops_per_sec": 4_000_000},
            "memory": {"write_gbps": 3.0, "read_gbps": 5.0},
            "thread_latency": {"avg_latency_us": 80.0, "p95_latency_us": 150.0},
            "gpu": {"available": False},
        }
        profile = get_or_tune(quick=True)
        assert profile.overall_score > 0

    def test_uses_existing(self, _tmp_tune_dir):
        p = TuneProfile(
            timestamp=datetime.now(timezone.utc).isoformat(),
            hardware_hash=hardware_fingerprint(),
            overall_score=88,
        )
        save_profile(p)
        profile = get_or_tune()
        assert profile.overall_score == 88


# ── Apply profile ─────────────────────────────────────────────────────────

class TestApplyProfile:
    def test_apply_sets_env(self, _tmp_tune_dir):
        p = TuneProfile(
            timestamp=datetime.now(timezone.utc).isoformat(),
            hardware_hash=hardware_fingerprint(),
            optimal_io_workers=12,
            optimal_cpu_workers=4,
            recommended_priority="NORMAL",
            recommended_energy="BALANCED",
        )
        save_profile(p)
        result = apply_profile(p)
        assert result["io_workers"] == 12
        assert result["cpu_workers"] == 4
        assert "priority" in result
        assert "energy" in result
        # Cleanup env
        os.environ.pop("PYACCELERATE_IO_WORKERS", None)
        os.environ.pop("PYACCELERATE_CPU_WORKERS", None)


# ── Profile summary ──────────────────────────────────────────────────────

class TestProfileSummary:
    def test_no_profile(self):
        text = profile_summary()
        assert "No tune profile found" in text

    def test_with_profile(self, _tmp_tune_dir):
        p = TuneProfile(
            timestamp="2026-01-01T00:00:00+00:00",
            hardware_hash="abc123",
            overall_score=75,
            cpu_score=80,
            memory_score=60,
            gpu_score=70,
            optimal_io_workers=16,
            optimal_cpu_workers=8,
        )
        save_profile(p)
        text = profile_summary()
        assert "75/100" in text
        assert "abc123" in text
        assert "16" in text
