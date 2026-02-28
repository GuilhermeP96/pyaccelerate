"""Tests for pyaccelerate.virt module."""

from pyaccelerate.virt import VirtInfo, detect


class TestVirtInfo:
    def test_defaults(self):
        vi = VirtInfo()
        assert vi.any_hw_virt is False
        assert vi.summary_parts() == []

    def test_any_hw_virt(self):
        vi = VirtInfo(vtx_enabled=True)
        assert vi.any_hw_virt is True

    def test_summary_parts(self):
        vi = VirtInfo(vtx_enabled=True, docker_available=True)
        parts = vi.summary_parts()
        assert "VT-x/AMD-V" in parts
        assert "Docker" in parts


class TestDetect:
    def test_returns_virtinfo(self):
        vi = detect()
        assert isinstance(vi, VirtInfo)

    def test_is_cached(self):
        a = detect()
        b = detect()
        assert a is b

    def test_platform_name(self):
        vi = detect()
        assert vi.platform_name != ""
