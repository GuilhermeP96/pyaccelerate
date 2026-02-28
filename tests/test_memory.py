"""Tests for pyaccelerate.memory module."""

from pyaccelerate.memory import (
    Pressure,
    get_pressure,
    get_stats,
    clamp_workers,
    BufferPool,
)


class TestPressure:
    def test_get_pressure(self):
        p = get_pressure()
        assert isinstance(p, Pressure)

    def test_get_stats(self):
        stats = get_stats()
        assert isinstance(stats, dict)


class TestClampWorkers:
    def test_no_clamp_at_low(self):
        # Can't control actual pressure, but we can test the API contract
        result = clamp_workers(16, floor=1)
        assert 1 <= result <= 16


class TestBufferPool:
    def test_acquire_release(self):
        pool = BufferPool(buffer_size=1024, max_buffers=4)
        buf = pool.acquire()
        assert len(buf) == 1024
        pool.release(buf)
        assert pool.stats["pooled"] == 1

    def test_reuse(self):
        pool = BufferPool(buffer_size=1024, max_buffers=2)
        buf1 = pool.acquire()
        pool.release(buf1)
        buf2 = pool.acquire()
        assert buf1 is buf2  # should be the same object

    def test_max_buffers(self):
        pool = BufferPool(buffer_size=64, max_buffers=2)
        bufs = [pool.acquire() for _ in range(5)]
        for b in bufs:
            pool.release(b)
        assert pool.stats["pooled"] == 2  # capped at max

    def test_clear(self):
        pool = BufferPool(buffer_size=64, max_buffers=4)
        bufs = [pool.acquire() for _ in range(4)]
        for b in bufs:
            pool.release(b)
        pool.clear()
        assert pool.stats["pooled"] == 0
