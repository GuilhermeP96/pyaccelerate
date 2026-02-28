"""Tests for pyaccelerate.npu subpackage."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════════════
#  NPU Detector Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNPUDevice:
    """Tests for the NPUDevice dataclass."""

    def test_default_device(self):
        from pyaccelerate.npu.detector import NPUDevice
        d = NPUDevice()
        assert d.name == ""
        assert d.vendor == ""
        assert d.backend == ""
        assert d.tops == 0.0
        assert d.usable is False
        assert d.score >= 1

    def test_usable_when_backend_set(self):
        from pyaccelerate.npu.detector import NPUDevice
        d = NPUDevice(name="Intel NPU", backend="openvino", tops=10.0)
        assert d.usable is True

    def test_usable_false_when_no_backend(self):
        from pyaccelerate.npu.detector import NPUDevice
        d = NPUDevice(name="Intel NPU", backend="none", tops=10.0)
        assert d.usable is False

    def test_score_includes_tops_and_backend(self):
        from pyaccelerate.npu.detector import NPUDevice
        d1 = NPUDevice(backend="openvino", tops=10.0)
        d2 = NPUDevice(backend="onnxrt-dml", tops=10.0)
        # openvino has higher bonus
        assert d1.score > d2.score

    def test_short_label(self):
        from pyaccelerate.npu.detector import NPUDevice
        d = NPUDevice(name="Intel NPU", backend="openvino", tops=13.0)
        label = d.short_label()
        assert "Intel NPU" in label
        assert "openvino" in label
        assert "13.0 TOPS" in label

    def test_as_dict(self):
        from pyaccelerate.npu.detector import NPUDevice
        d = NPUDevice(name="Test", vendor="Intel", backend="openvino", tops=10.0)
        dd = d.as_dict()
        assert dd["name"] == "Test"
        assert dd["vendor"] == "Intel"
        assert dd["backend"] == "openvino"
        assert dd["usable"] == "True"


class TestNPUDetection:
    """Tests for NPU detection functions."""

    def setup_method(self):
        from pyaccelerate.npu.detector import reset_cache
        reset_cache()

    def test_detect_all_returns_list(self):
        from pyaccelerate.npu.detector import detect_all
        result = detect_all()
        assert isinstance(result, list)

    def test_detect_all_cached(self):
        from pyaccelerate.npu.detector import detect_all
        r1 = detect_all()
        r2 = detect_all()
        assert r1 is r2

    def test_reset_cache(self):
        from pyaccelerate.npu.detector import detect_all, reset_cache
        r1 = detect_all()
        reset_cache()
        r2 = detect_all()
        # Fresh detection, not same object
        assert r1 is not r2

    def test_npu_available_returns_bool(self):
        from pyaccelerate.npu.detector import npu_available
        result = npu_available()
        assert isinstance(result, bool)

    def test_best_npu_returns_device_or_none(self):
        from pyaccelerate.npu.detector import best_npu, NPUDevice
        result = best_npu()
        assert result is None or isinstance(result, NPUDevice)

    def test_get_npu_info(self):
        from pyaccelerate.npu.detector import get_npu_info
        info = get_npu_info()
        assert isinstance(info, dict)
        assert "available" in info or "backend" in info

    def test_get_all_npus_info(self):
        from pyaccelerate.npu.detector import get_all_npus_info
        info = get_all_npus_info()
        assert isinstance(info, list)


class TestVendorDetection:
    """Tests for vendor/TOPS heuristics."""

    def test_vendor_intel(self):
        from pyaccelerate.npu.detector import _vendor_from_name
        assert _vendor_from_name("Intel Meteor Lake") == "Intel"
        assert _vendor_from_name("Arrow Lake NPU") == "Intel"

    def test_vendor_amd(self):
        from pyaccelerate.npu.detector import _vendor_from_name
        assert _vendor_from_name("AMD Ryzen AI") == "AMD"
        assert _vendor_from_name("XDNA Processor") == "AMD"

    def test_vendor_qualcomm(self):
        from pyaccelerate.npu.detector import _vendor_from_name
        assert _vendor_from_name("Qualcomm Hexagon") == "Qualcomm"
        assert _vendor_from_name("Snapdragon X Elite") == "Qualcomm"

    def test_vendor_apple(self):
        from pyaccelerate.npu.detector import _vendor_from_name
        assert _vendor_from_name("Apple Neural Engine") == "Apple"

    def test_vendor_unknown(self):
        from pyaccelerate.npu.detector import _vendor_from_name
        assert _vendor_from_name("Some Unknown Device") == "unknown"

    def test_estimate_tops_known(self):
        from pyaccelerate.npu.detector import _estimate_tops
        assert _estimate_tops("Intel Meteor Lake NPU") == 10.0
        assert _estimate_tops("Snapdragon X Elite") == 45.0
        assert _estimate_tops("Apple M4 Pro") == 38.0

    def test_estimate_tops_unknown(self):
        from pyaccelerate.npu.detector import _estimate_tops
        assert _estimate_tops("Unknown Device") == 0.0


class TestGetInstallHint:
    """Tests for install hint generation."""

    def test_hint_returns_string(self):
        from pyaccelerate.npu.detector import get_install_hint, reset_cache
        reset_cache()
        hint = get_install_hint()
        assert isinstance(hint, str)


# ═══════════════════════════════════════════════════════════════════════════
#  ONNX Runtime Bridge Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOnnxRtBridge:
    """Tests for the ONNX Runtime NPU bridge (mocked)."""

    def test_list_all_eps_no_ort(self):
        from pyaccelerate.npu.onnx_rt import list_all_eps
        # May or may not be installed — just verify it returns a list
        result = list_all_eps()
        assert isinstance(result, list)

    def test_list_npu_eps(self):
        from pyaccelerate.npu.onnx_rt import list_npu_eps
        result = list_npu_eps()
        assert isinstance(result, list)

    def test_best_ep(self):
        from pyaccelerate.npu.onnx_rt import best_ep
        result = best_ep()
        assert result is None or isinstance(result, str)

    def test_onnxrt_available(self):
        from pyaccelerate.npu.onnx_rt import onnxrt_available
        result = onnxrt_available()
        assert isinstance(result, bool)

    def test_npu_ep_available(self):
        from pyaccelerate.npu.onnx_rt import npu_ep_available
        result = npu_ep_available()
        assert isinstance(result, bool)

    def test_build_provider_options_openvino(self):
        from pyaccelerate.npu.onnx_rt import _build_provider_options
        opts = _build_provider_options("OpenVINOExecutionProvider")
        assert opts["device_type"] == "NPU"

    def test_build_provider_options_dml(self):
        from pyaccelerate.npu.onnx_rt import _build_provider_options
        opts = _build_provider_options("DmlExecutionProvider")
        assert "device_id" in opts

    def test_build_provider_options_qnn(self):
        from pyaccelerate.npu.onnx_rt import _build_provider_options
        opts = _build_provider_options("QNNExecutionProvider")
        assert isinstance(opts, dict)

    def test_build_provider_options_unknown(self):
        from pyaccelerate.npu.onnx_rt import _build_provider_options
        opts = _build_provider_options("SomeUnknownEP")
        assert opts == {}


# ═══════════════════════════════════════════════════════════════════════════
#  OpenVINO Helpers Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOpenVINOHelpers:
    """Tests for OpenVINO helpers (mocked — no real hardware needed)."""

    def test_available_returns_bool(self):
        from pyaccelerate.npu.openvino import available
        result = available()
        assert isinstance(result, bool)

    def test_list_devices_returns_list(self):
        from pyaccelerate.npu.openvino import list_devices
        result = list_devices()
        assert isinstance(result, list)

    def test_get_npu_properties_returns_dict(self):
        from pyaccelerate.npu.openvino import get_npu_properties
        result = get_npu_properties()
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
#  Inference API Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestInferenceSession:
    """Tests for the unified InferenceSession."""

    def test_backend_enum(self):
        from pyaccelerate.npu.inference import Backend
        assert Backend.OPENVINO.value == "openvino"
        assert Backend.ONNXRT.value == "onnxrt"
        assert Backend.CPU.value == "cpu"

    def test_session_init_fails_gracefully(self):
        """Without ONNX Runtime or OpenVINO, session creation should raise."""
        from pyaccelerate.npu.inference import InferenceSession
        # Try with a non-existent model — should fail either because no
        # backend or because file doesn't exist
        with pytest.raises(Exception):
            InferenceSession("nonexistent_model.onnx")


# ═══════════════════════════════════════════════════════════════════════════
#  Engine NPU Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEngineNPUIntegration:
    """Tests for NPU support in Engine."""

    def test_engine_has_npus_attr(self):
        from pyaccelerate.engine import Engine
        engine = Engine()
        assert hasattr(engine, 'npus')
        assert isinstance(engine.npus, list)

    def test_engine_npu_enabled_flag(self):
        from pyaccelerate.engine import Engine
        engine = Engine(npu_enabled=False)
        assert engine.npu_enabled is False
        assert engine.npus == []

    def test_engine_best_npu(self):
        from pyaccelerate.engine import Engine
        engine = Engine()
        # best_npu is None or NPUDevice
        result = engine.best_npu
        from pyaccelerate.npu.detector import NPUDevice
        assert result is None or isinstance(result, NPUDevice)

    def test_engine_summary_includes_npu(self):
        from pyaccelerate.engine import Engine
        engine = Engine()
        summary = engine.summary()
        assert "NPU" in summary

    def test_engine_status_line_includes_npu(self):
        from pyaccelerate.engine import Engine
        engine = Engine()
        status = engine.status_line()
        assert "NPU" in status

    def test_engine_as_dict_includes_npu(self):
        from pyaccelerate.engine import Engine
        engine = Engine()
        d = engine.as_dict()
        assert "npu" in d
        assert "enabled" in d["npu"]
        assert "devices" in d["npu"]

    def test_engine_set_npu_enabled(self):
        from pyaccelerate.engine import Engine
        engine = Engine(npu_enabled=False)
        assert engine.npus == []
        engine.set_npu_enabled(True)
        assert engine.npu_enabled is True
        # npus should be detected now
        assert isinstance(engine.npus, list)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI NPU Subcommand Test
# ═══════════════════════════════════════════════════════════════════════════

class TestCLINPU:
    """Test the CLI 'npu' subcommand works."""

    def test_cli_npu(self, capsys):
        from pyaccelerate.cli import main
        main(["npu"])
        out = capsys.readouterr().out
        # Should produce some output (even if "No NPU detected")
        assert "NPU" in out or "npu" in out.lower()

    def test_cli_info_includes_npu(self, capsys):
        from pyaccelerate.cli import main
        main(["info"])
        out = capsys.readouterr().out
        assert "NPU" in out
