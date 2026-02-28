"""Tests for pyaccelerate.gpu module."""

import pytest

from pyaccelerate.gpu.detector import (
    GPUDevice,
    detect_all,
    gpu_available,
    best_gpu,
    get_gpu_info,
    get_all_gpus_info,
    get_install_hint,
    _vendor_from_name,
)


class TestGPUDevice:
    def test_score_discrete_bonus(self):
        d = GPUDevice(memory_bytes=4 * 1024**3, compute_units=30, is_discrete=True)
        i = GPUDevice(memory_bytes=1 * 1024**3, compute_units=16, is_discrete=False)
        assert d.score > i.score

    def test_memory_gb(self):
        d = GPUDevice(memory_bytes=8 * 1024**3)
        assert abs(d.memory_gb - 8.0) < 0.01

    def test_usable(self):
        assert GPUDevice(backend="cuda").usable is True
        assert GPUDevice(backend="none").usable is False

    def test_short_label(self):
        d = GPUDevice(name="RTX 4090", backend="cuda", memory_bytes=24 * 1024**3)
        label = d.short_label()
        assert "RTX 4090" in label
        assert "CUDA" in label

    def test_as_dict(self):
        d = GPUDevice(name="test", backend="cuda", vendor="NVIDIA")
        info = d.as_dict()
        assert info["name"] == "test"
        assert info["backend"] == "cuda"


class TestVendorDetection:
    def test_nvidia(self):
        vendor, discrete = _vendor_from_name("NVIDIA GeForce RTX 4090")
        assert vendor == "NVIDIA"
        assert discrete is True

    def test_amd(self):
        vendor, discrete = _vendor_from_name("AMD Radeon RX 7900")
        assert vendor == "AMD"
        assert discrete is True

    def test_intel_integrated(self):
        vendor, discrete = _vendor_from_name("Intel UHD 770")
        assert vendor == "Intel"
        assert discrete is False

    def test_intel_arc(self):
        vendor, discrete = _vendor_from_name("Intel Arc A770")
        assert vendor == "Intel"
        assert discrete is True

    def test_unknown(self):
        vendor, discrete = _vendor_from_name("SomeBrand GPU 3000")
        assert vendor == "unknown"


class TestDetectAll:
    def test_returns_list(self):
        gpus = detect_all()
        assert isinstance(gpus, list)

    def test_sorted_by_score(self):
        gpus = detect_all()
        if len(gpus) > 1:
            assert gpus[0].score >= gpus[1].score


class TestPublicAPI:
    def test_gpu_available_bool(self):
        result = gpu_available()
        assert isinstance(result, bool)

    def test_get_gpu_info_dict(self):
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "available" in info
        assert "backend" in info

    def test_get_all_gpus_info_list(self):
        infos = get_all_gpus_info()
        assert isinstance(infos, list)

    def test_install_hint(self):
        hint = get_install_hint()
        assert isinstance(hint, str)
