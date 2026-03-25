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
    _classify_nvidia,
    _classify_amd,
)


class TestGPUDevice:
    def test_score_discrete_bonus(self):
        d = GPUDevice(memory_bytes=4 * 1024**3, compute_units=30, is_discrete=True)
        i = GPUDevice(memory_bytes=1 * 1024**3, compute_units=16, is_discrete=False)
        assert d.score > i.score

    def test_score_tensor_bonus(self):
        no_tensor = GPUDevice(memory_bytes=8 * 1024**3, compute_units=30, is_discrete=True)
        with_tensor = GPUDevice(memory_bytes=8 * 1024**3, compute_units=30, is_discrete=True, has_tensor=True)
        assert with_tensor.score > no_tensor.score

    def test_score_rt_bonus(self):
        no_rt = GPUDevice(memory_bytes=8 * 1024**3, compute_units=30, is_discrete=True)
        with_rt = GPUDevice(memory_bytes=8 * 1024**3, compute_units=30, is_discrete=True, has_raytracing=True)
        assert with_rt.score > no_rt.score

    def test_memory_gb(self):
        d = GPUDevice(memory_bytes=8 * 1024**3)
        assert abs(d.memory_gb - 8.0) < 0.01

    def test_shared_memory(self):
        d = GPUDevice(memory_bytes=6 * 1024**3, shared_memory_bytes=28 * 1024**3)
        assert abs(d.shared_memory_gb - 28.0) < 0.1
        assert abs(d.total_memory_gb - 34.0) < 0.1

    def test_usable(self):
        assert GPUDevice(backend="cuda").usable is True
        assert GPUDevice(backend="none").usable is False

    def test_short_label_with_arch(self):
        d = GPUDevice(name="RTX 4090", backend="cuda",
                      memory_bytes=24 * 1024**3, architecture="Ada Lovelace")
        label = d.short_label()
        assert "RTX 4090" in label
        assert "CUDA" in label
        assert "Ada Lovelace" in label

    def test_as_dict_extended(self):
        d = GPUDevice(
            name="GeForce GTX 1660 SUPER", backend="cuda", vendor="NVIDIA",
            architecture="Turing", cuda_capability="7.5",
            cuda_cores=1408, has_nvenc=True, has_nvdec=True,
            memory_type="GDDR6", memory_bus_width=192,
        )
        info = d.as_dict()
        assert info["name"] == "GeForce GTX 1660 SUPER"
        assert info["architecture"] == "Turing"
        assert info["cuda_capability"] == "7.5"
        assert info["cuda_cores"] == "1408"
        assert info["has_hw_encode"] == "True"
        assert info["memory_type"] == "GDDR6"
        assert info["memory_bus_width"] == "192-bit"
        assert "features" in info

    def test_features_list(self):
        d = GPUDevice(
            backend="cuda", has_tensor=True, has_raytracing=True,
            has_nvenc=True, has_nvdec=True,
            cuda_capability="8.6", architecture="Ampere", copy_engines=2,
        )
        feats = d.features
        assert "compute" in feats
        assert "tensor" in feats
        assert "raytracing" in feats
        assert "hw_encode" in feats
        assert "hw_decode" in feats
        assert "cuda_8.6" in feats
        assert "ampere" in feats

    def test_features_gtx_no_tensor(self):
        d = GPUDevice(
            backend="cuda", has_tensor=False, has_raytracing=False,
            has_nvenc=True, has_nvdec=True,
            cuda_capability="7.5", architecture="Turing",
        )
        feats = d.features
        assert "compute" in feats
        assert "tensor" not in feats
        assert "raytracing" not in feats
        assert "hw_encode" in feats


class TestNvidiaClassification:
    def test_gtx_1660_super(self):
        info = _classify_nvidia("NVIDIA GeForce GTX 1660 SUPER", 7, 5, 22)
        assert info["architecture"] == "Turing"
        assert info["cuda_capability"] == "7.5"
        assert info["cuda_cores"] == 22 * 64  # 1408
        assert info["has_tensor"] is False  # GTX = no tensor
        assert info["has_raytracing"] is False  # GTX = no RT
        assert info["has_nvenc"] is True
        assert info["has_nvdec"] is True
        assert info["memory_type"] == "GDDR6"

    def test_rtx_2060(self):
        info = _classify_nvidia("NVIDIA GeForce RTX 2060", 7, 5, 30)
        assert info["architecture"] == "Turing"
        assert info["has_tensor"] is True
        assert info["tensor_cores"] == 30 * 8
        assert info["has_raytracing"] is True
        assert info["rt_cores"] == 30

    def test_rtx_3080(self):
        info = _classify_nvidia("NVIDIA GeForce RTX 3080", 8, 6, 68)
        assert info["architecture"] == "Ampere"
        assert info["cuda_cores"] == 68 * 128  # 8704
        assert info["has_tensor"] is True
        assert info["tensor_cores"] == 68 * 4
        assert info["has_raytracing"] is True
        assert info["rt_cores"] == 68

    def test_rtx_4090(self):
        info = _classify_nvidia("NVIDIA GeForce RTX 4090", 8, 9, 128)
        assert info["architecture"] == "Ada Lovelace"
        assert info["cuda_cores"] == 128 * 128  # 16384
        assert info["has_tensor"] is True
        assert info["has_raytracing"] is True

    def test_a100(self):
        info = _classify_nvidia("NVIDIA A100 80GB", 8, 0, 108)
        assert info["architecture"] == "Ampere"
        assert info["has_tensor"] is True
        assert info["has_raytracing"] is False  # data-center
        assert info["memory_type"] == "HBM2e"

    def test_rtx_5090(self):
        info = _classify_nvidia("NVIDIA GeForce RTX 5090", 10, 0, 170)
        assert info["architecture"] == "Blackwell"
        assert info["has_tensor"] is True
        assert info["has_raytracing"] is True


class TestAmdClassification:
    def test_rx_7900(self):
        info = _classify_amd("AMD Radeon RX 7900 XTX")
        assert info["architecture"] == "RDNA 3"
        assert info["has_raytracing"] is True
        assert info["has_nvenc"] is True  # VCN encode

    def test_rx_6600(self):
        info = _classify_amd("AMD Radeon RX 6600")
        assert info["architecture"] == "RDNA 2"
        assert info["has_raytracing"] is True

    def test_rx_580(self):
        info = _classify_amd("AMD Radeon RX 580")
        assert info["architecture"] == "GCN 4"
        assert info["has_raytracing"] is False

    def test_rx_5700(self):
        info = _classify_amd("AMD Radeon RX 5700 XT")
        assert info["architecture"] == "RDNA"
        assert info["has_raytracing"] is False

    def test_instinct_mi300(self):
        info = _classify_amd("AMD Instinct MI300X")
        assert info["architecture"] == "CDNA 3"

    def test_unknown_amd(self):
        info = _classify_amd("AMD Unknown Model")
        assert info == {}


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
