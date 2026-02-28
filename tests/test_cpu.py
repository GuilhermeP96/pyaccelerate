"""Tests for pyaccelerate.cpu module."""

from pyaccelerate.cpu import (
    CPUInfo,
    detect,
    recommend_workers,
    recommend_io_workers,
    _parse_cpulist_count,
)


class TestCPUDetect:
    def test_detect_returns_cpuinfo(self):
        info = detect()
        assert isinstance(info, CPUInfo)
        assert info.logical_cores >= 1
        assert info.physical_cores >= 1

    def test_detect_is_cached(self):
        a = detect()
        b = detect()
        assert a is b

    def test_smt_ratio(self):
        info = detect()
        assert info.smt_ratio >= 1.0

    def test_short_label(self):
        info = detect()
        label = info.short_label()
        assert "T" in label
        assert "C" in label


class TestRecommendWorkers:
    def test_io_bound_returns_positive(self):
        n = recommend_workers(io_bound=True)
        assert n >= 1

    def test_cpu_bound_returns_positive(self):
        n = recommend_workers(io_bound=False)
        assert n >= 1

    def test_io_bound_higher_than_cpu(self):
        io_n = recommend_workers(io_bound=True)
        cpu_n = recommend_workers(io_bound=False)
        assert io_n >= cpu_n

    def test_io_workers_tuple(self):
        pull, push = recommend_io_workers()
        assert pull >= 2
        assert push >= 2
        assert pull >= push  # pull cap is higher


class TestParseCpulist:
    def test_single(self):
        assert _parse_cpulist_count("0") == 1

    def test_range(self):
        assert _parse_cpulist_count("0-3") == 4

    def test_mixed(self):
        assert _parse_cpulist_count("0-3,5,7-9") == 8

    def test_empty(self):
        assert _parse_cpulist_count("") == 0
