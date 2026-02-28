"""Tests for pyaccelerate.engine module."""

from pyaccelerate.engine import Engine


class TestEngine:
    def test_create(self):
        engine = Engine()
        assert engine.cpu.logical_cores >= 1

    def test_summary(self):
        engine = Engine()
        summary = engine.summary()
        assert "PyAccelerate" in summary
        assert "CPU" in summary

    def test_status_line(self):
        engine = Engine()
        line = engine.status_line()
        assert "CPU" in line

    def test_as_dict(self):
        engine = Engine()
        d = engine.as_dict()
        assert "cpu" in d
        assert "gpu" in d
        assert "memory" in d
        assert "pools" in d

    def test_submit(self):
        engine = Engine()
        fut = engine.submit(lambda x: x + 1, 41)
        assert fut.result(timeout=5) == 42

    def test_run_parallel(self):
        engine = Engine()
        results = []
        count = engine.run_parallel(
            lambda x: results.append(x),
            [(i,) for i in range(10)],
            max_concurrent=4,
        )
        assert count == 10

    def test_shutdown(self):
        engine = Engine()
        engine.shutdown(wait_for=True)
