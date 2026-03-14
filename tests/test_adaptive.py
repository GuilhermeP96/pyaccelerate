"""Tests for pyaccelerate.adaptive module."""

import time

from pyaccelerate.adaptive import (
    AdaptiveConfig,
    AdaptiveScheduler,
    AdaptiveSnapshot,
    LatencyTracker,
    get_cpu_pressure,
)


class TestLatencyTracker:
    def test_record_and_percentiles(self):
        lt = LatencyTracker(window=100)
        for i in range(100):
            lt.record(float(i))
        assert lt.count == 100
        assert lt.p50 >= 40
        assert lt.p95 >= 90
        assert lt.p99 >= 95

    def test_empty_tracker(self):
        lt = LatencyTracker()
        assert lt.p50 == 0.0
        assert lt.p95 == 0.0
        assert lt.count == 0

    def test_reset(self):
        lt = LatencyTracker()
        lt.record(100.0)
        lt.record(200.0)
        assert lt.count == 2
        lt.reset()
        assert lt.count == 0

    def test_ring_buffer_overflow(self):
        lt = LatencyTracker(window=10)
        for i in range(20):
            lt.record(float(i))
        assert lt.count == 10  # capped at window size


class TestCPUPressure:
    def test_returns_float(self):
        cpu = get_cpu_pressure()
        assert isinstance(cpu, float)
        assert 0.0 <= cpu <= 100.0


class TestAdaptiveConfig:
    def test_defaults(self):
        cfg = AdaptiveConfig()
        assert cfg.min_workers >= 2
        assert cfg.max_workers > 0
        assert cfg.cooldown_seconds > 0

    def test_custom(self):
        cfg = AdaptiveConfig(min_workers=1, max_workers=8, cooldown_seconds=0.5)
        assert cfg.min_workers == 1
        assert cfg.max_workers == 8


class TestAdaptiveScheduler:
    def test_create_and_start(self):
        sched = AdaptiveScheduler(num_workers=2)
        sched.start()
        assert sched._running
        sched.shutdown()

    def test_context_manager(self):
        with AdaptiveScheduler(num_workers=2) as sched:
            assert sched._running

    def test_submit(self):
        with AdaptiveScheduler(num_workers=2) as sched:
            fut = sched.submit(lambda x: x + 1, 41)
            assert fut.result(timeout=5) == 42

    def test_map(self):
        with AdaptiveScheduler(num_workers=2) as sched:
            results = sched.map(lambda x: x ** 2, [(i,) for i in range(10)])
            assert results == [i ** 2 for i in range(10)]

    def test_snapshot(self):
        with AdaptiveScheduler(num_workers=2) as sched:
            # Submit some work to populate latency tracker
            futures = [sched.submit(lambda x: x, i) for i in range(10)]
            for f in futures:
                f.result(timeout=5)
            time.sleep(0.1)
            snap = sched.snapshot()
            assert isinstance(snap, AdaptiveSnapshot)
            assert snap.workers >= 2
            assert snap.memory_pressure in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_adaptive_adjusts_under_load(self):
        """The adaptive scheduler should remain stable under normal load."""
        cfg = AdaptiveConfig(
            min_workers=2,
            max_workers=8,
            cooldown_seconds=0.2,
            poll_interval_seconds=0.1,
        )
        with AdaptiveScheduler(config=cfg, num_workers=2) as sched:
            # Submit moderate workload
            futures = [sched.submit(lambda x: x, i) for i in range(50)]
            for f in futures:
                f.result(timeout=10)
            snap = sched.snapshot()
            assert snap.workers >= cfg.min_workers
            assert snap.workers <= cfg.max_workers
