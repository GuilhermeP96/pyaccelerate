"""Tests for pyaccelerate.profiler module."""

import time

from pyaccelerate.profiler import timed, profile_memory, Timer, timer, Tracker


class TestTimed:
    def test_decorator_preserves_result(self):
        @timed()
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_decorator_preserves_name(self):
        @timed()
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


class TestProfileMemory:
    def test_decorator_preserves_result(self):
        @profile_memory()
        def mul(a, b):
            return a * b

        assert mul(3, 4) == 12


class TestTimer:
    def test_context_manager(self):
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.005

    def test_functional_timer(self):
        with timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.005


class TestTracker:
    def test_measure(self):
        tracker = Tracker("test")
        for _ in range(10):
            with tracker.measure():
                time.sleep(0.001)
        assert tracker.count == 10
        assert tracker.total > 0
        assert tracker.mean > 0

    def test_record(self):
        tracker = Tracker()
        tracker.record(1.0)
        tracker.record(2.0)
        tracker.record(3.0)
        assert tracker.count == 3
        assert tracker.mean == 2.0
        assert tracker.min_time == 1.0
        assert tracker.max_time == 3.0

    def test_percentiles(self):
        tracker = Tracker()
        for i in range(100):
            tracker.record(float(i))
        assert tracker.p50 == 50.0
        assert tracker.p95 == 95.0

    def test_summary(self):
        tracker = Tracker("ops")
        tracker.record(0.5)
        summary = tracker.summary()
        assert "ops" in summary
        assert "1 calls" in summary

    def test_reset(self):
        tracker = Tracker()
        tracker.record(1.0)
        tracker.reset()
        assert tracker.count == 0
