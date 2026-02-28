"""Tests for pyaccelerate.android module — Android/Termux/ARM detection."""

from __future__ import annotations

import platform
from unittest.mock import patch, MagicMock, mock_open

import pytest


# ═══════════════════════════════════════════════════════════════════════════
#  is_android / is_termux / is_arm
# ═══════════════════════════════════════════════════════════════════════════

class TestPlatformDetection:
    """Test Android, Termux and ARM detection helpers."""

    def setup_method(self):
        # Reset cached globals before each test
        import pyaccelerate.android as mod
        mod._is_android = None
        mod._is_termux = None
        mod._device_info = None

    def test_is_arm_aarch64(self):
        from pyaccelerate.android import is_arm
        with patch("platform.machine", return_value="aarch64"):
            assert is_arm() is True

    def test_is_arm_arm64(self):
        from pyaccelerate.android import is_arm
        with patch("platform.machine", return_value="arm64"):
            assert is_arm() is True

    def test_is_arm_armv7l(self):
        from pyaccelerate.android import is_arm
        with patch("platform.machine", return_value="armv7l"):
            assert is_arm() is True

    def test_is_arm_x86_false(self):
        from pyaccelerate.android import is_arm
        with patch("platform.machine", return_value="x86_64"):
            assert is_arm() is False

    @patch.dict("os.environ", {"ANDROID_ROOT": "/system"}, clear=False)
    def test_is_android_env(self):
        from pyaccelerate.android import is_android
        import pyaccelerate.android as mod
        mod._is_android = None
        assert is_android() is True

    @patch.dict("os.environ", {"TERMUX_VERSION": "0.118"}, clear=False)
    def test_is_termux_env(self):
        from pyaccelerate.android import is_termux
        import pyaccelerate.android as mod
        mod._is_termux = None
        assert is_termux() is True

    @patch.dict("os.environ", {}, clear=True)
    @patch("os.path.isdir", return_value=False)
    @patch("os.path.isfile", return_value=False)
    @patch("platform.platform", return_value="Windows-10")
    @patch("platform.machine", return_value="x86_64")
    def test_is_android_false_on_desktop(self, *mocks):
        from pyaccelerate.android import is_android
        import pyaccelerate.android as mod
        mod._is_android = None
        assert is_android() is False

    @patch.dict("os.environ", {}, clear=True)
    @patch("os.path.isdir", return_value=False)
    def test_is_termux_false_on_desktop(self, *mocks):
        from pyaccelerate.android import is_termux
        import pyaccelerate.android as mod
        mod._is_termux = None
        assert is_termux() is False


# ═══════════════════════════════════════════════════════════════════════════
#  SoC Database
# ═══════════════════════════════════════════════════════════════════════════

class TestSoCDatabase:
    """Test the SoC database and lookup logic."""

    def test_soc_info_fields(self):
        from pyaccelerate.android import SoCInfo
        soc = SoCInfo(
            name="Snapdragon 8 Gen 3", vendor="Qualcomm", cpu_arch="ARMv9.2",
            cpu_cores_big=1, cpu_cores_mid=3, cpu_cores_little=4,
            gpu_name="Adreno 750", gpu_cores=0, npu_name="Hexagon NPU",
            npu_tops=73.0, process_nm=4,
        )
        assert soc.name == "Snapdragon 8 Gen 3"
        assert soc.vendor == "Qualcomm"
        assert soc.npu_tops == 73.0

    def test_soc_database_not_empty(self):
        from pyaccelerate.android import _SOC_DATABASE
        assert len(_SOC_DATABASE) >= 25

    def test_lookup_soc_by_board(self):
        from pyaccelerate.android import _lookup_soc
        soc = _lookup_soc("sm8650", "", "")
        assert soc is not None
        assert "Snapdragon 8 Gen 3" in soc.name

    def test_lookup_soc_by_hardware(self):
        from pyaccelerate.android import _lookup_soc
        soc = _lookup_soc("", "sm8550", "")
        assert soc is not None
        assert "Snapdragon 8 Gen 2" in soc.name

    def test_lookup_soc_by_chipset(self):
        from pyaccelerate.android import _lookup_soc
        soc = _lookup_soc("", "", "mt6989")
        assert soc is not None
        assert "Dimensity 9300" in soc.name

    def test_lookup_soc_unknown(self):
        from pyaccelerate.android import _lookup_soc
        soc = _lookup_soc("unknown_board", "unknown_hw", "unknown_chip")
        assert soc is None

    def test_all_socs_have_gpu(self):
        from pyaccelerate.android import _SOC_DATABASE
        for key, soc in _SOC_DATABASE.items():
            if key != "ums9230":  # Budget Unisoc has minimal GPU
                assert soc.gpu_name, f"SoC {key} missing GPU name"

    def test_all_qualcomm_have_hexagon(self):
        from pyaccelerate.android import _SOC_DATABASE
        for key, soc in _SOC_DATABASE.items():
            if soc.vendor == "Qualcomm":
                assert "Hexagon" in soc.npu_name or "hexagon" in soc.npu_name.lower(), \
                    f"Qualcomm SoC {key} should have Hexagon NPU"

    def test_lookup_exynos(self):
        from pyaccelerate.android import _lookup_soc
        soc = _lookup_soc("s5e9925", "", "")
        assert soc is not None
        assert soc.vendor == "Samsung"
        assert "Xclipse" in soc.gpu_name

    def test_lookup_tensor(self):
        from pyaccelerate.android import _lookup_soc
        soc = _lookup_soc("zuma", "", "")
        assert soc is not None
        assert soc.vendor == "Google"
        assert "Tensor G3" in soc.name

    def test_lookup_dimensity(self):
        from pyaccelerate.android import _lookup_soc
        soc = _lookup_soc("mt6985", "", "")
        assert soc is not None
        assert soc.vendor == "MediaTek"
        assert "9200" in soc.name

    def test_lookup_kirin(self):
        from pyaccelerate.android import _lookup_soc
        soc = _lookup_soc("kirin9000", "", "")
        assert soc is not None
        assert soc.vendor == "HiSilicon"


# ═══════════════════════════════════════════════════════════════════════════
#  ARM Core Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestARMCoreDetection:
    """Test ARM core identification from /proc/cpuinfo."""

    SAMPLE_CPUINFO = """\
processor\t: 0
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x1
CPU part\t: 0xd44
CPU revision\t: 1

processor\t: 1
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x1
CPU part\t: 0xd44
CPU revision\t: 1

processor\t: 2
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x1
CPU part\t: 0xd05
CPU revision\t: 0

processor\t: 3
CPU implementer\t: 0x41
CPU architecture: 8
CPU variant\t: 0x1
CPU part\t: 0xd05
CPU revision\t: 0

"""

    def test_arm_part_names_populated(self):
        from pyaccelerate.android import _ARM_PART_NAMES
        assert len(_ARM_PART_NAMES) >= 30
        assert _ARM_PART_NAMES.get("0xd44") == "Cortex-X1"
        assert _ARM_PART_NAMES.get("0xd05") == "Cortex-A55"
        assert _ARM_PART_NAMES.get("0xd0b") == "Cortex-A76"

    def test_arm_implementers_populated(self):
        from pyaccelerate.android import _ARM_IMPLEMENTERS
        assert _ARM_IMPLEMENTERS.get("0x41") == "ARM"
        assert _ARM_IMPLEMENTERS.get("0x51") == "Qualcomm"
        assert _ARM_IMPLEMENTERS.get("0x53") == "Samsung"

    @patch("builtins.open", mock_open(read_data=SAMPLE_CPUINFO))
    @patch("pathlib.Path.read_text")
    def test_detect_arm_cores(self, mock_read):
        mock_read.return_value = self.SAMPLE_CPUINFO
        from pyaccelerate.android import detect_arm_cores
        cores = detect_arm_cores()
        assert len(cores) == 4
        # First 2 should be Cortex-X1
        assert cores[0]["name"] == "Cortex-X1"
        assert cores[1]["name"] == "Cortex-X1"
        # Last 2 should be Cortex-A55
        assert cores[2]["name"] == "Cortex-A55"
        assert cores[3]["name"] == "Cortex-A55"

    @patch("pathlib.Path.read_text")
    def test_detect_big_little(self, mock_read):
        mock_read.return_value = self.SAMPLE_CPUINFO
        from pyaccelerate.android import detect_big_little
        clusters = detect_big_little()
        assert "Cortex-X1" in clusters
        assert "Cortex-A55" in clusters
        assert len(clusters["Cortex-X1"]) == 2
        assert len(clusters["Cortex-A55"]) == 2

    @patch("pathlib.Path.read_text")
    def test_get_arm_features(self, mock_read):
        mock_read.return_value = "Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 neon\n"
        from pyaccelerate.android import get_arm_features
        feats = get_arm_features()
        assert "neon" in feats
        assert "asimd" in feats
        assert "aes" in feats

    def test_get_arm_features_no_cpuinfo(self):
        from pyaccelerate.android import get_arm_features
        with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
            feats = get_arm_features()
            assert feats == []


# ═══════════════════════════════════════════════════════════════════════════
#  Thermal & Battery
# ═══════════════════════════════════════════════════════════════════════════

class TestThermalBattery:
    """Test thermal zone and battery reading."""

    def test_get_thermal_zones_returns_dict(self):
        from pyaccelerate.android import get_thermal_zones
        result = get_thermal_zones()
        assert isinstance(result, dict)

    def test_get_battery_info_returns_dict(self):
        from pyaccelerate.android import get_battery_info
        result = get_battery_info()
        assert isinstance(result, dict)

    def test_is_thermally_throttled_returns_bool(self):
        from pyaccelerate.android import is_thermally_throttled
        result = is_thermally_throttled()
        assert isinstance(result, bool)

    def test_get_cpu_freq_info_returns_dict(self):
        from pyaccelerate.android import get_cpu_freq_info
        result = get_cpu_freq_info()
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
#  CPU ARM integration tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCPUARMIntegration:
    """Test ARM-specific CPUInfo fields and methods."""

    def test_cpuinfo_arm_fields_default(self):
        from pyaccelerate.cpu import CPUInfo
        info = CPUInfo()
        assert info.is_arm is False
        assert info.arm_clusters == {}
        assert info.soc_name == ""
        assert info.is_android is False

    def test_cpuinfo_has_neon(self):
        from pyaccelerate.cpu import CPUInfo
        info = CPUInfo(flags=["neon", "asimd", "fp"])
        assert info.has_neon is True

    def test_cpuinfo_has_neon_false(self):
        from pyaccelerate.cpu import CPUInfo
        info = CPUInfo(flags=["sse", "avx"])
        assert info.has_neon is False

    def test_cpuinfo_has_sve(self):
        from pyaccelerate.cpu import CPUInfo
        info = CPUInfo(flags=["sve", "sve2", "neon"])
        assert info.has_sve is True

    def test_cpuinfo_has_sve_false(self):
        from pyaccelerate.cpu import CPUInfo
        info = CPUInfo(flags=["neon", "asimd"])
        assert info.has_sve is False

    def test_cpuinfo_big_cores(self):
        from pyaccelerate.cpu import CPUInfo
        info = CPUInfo(
            arm_clusters={"Cortex-X4": [0], "Cortex-A720": [1, 2, 3], "Cortex-A520": [4, 5, 6, 7]},
        )
        assert info.big_cores > 0

    def test_cpuinfo_little_cores(self):
        from pyaccelerate.cpu import CPUInfo
        info = CPUInfo(
            arm_clusters={"Cortex-X4": [0], "Cortex-A720": [1, 2, 3], "Cortex-A520": [4, 5, 6, 7]},
        )
        assert info.little_cores == 4

    def test_cpuinfo_no_clusters(self):
        from pyaccelerate.cpu import CPUInfo
        info = CPUInfo(arm_clusters={})
        assert info.big_cores == 0
        assert info.little_cores == 0

    def test_short_label_arm_clusters(self):
        from pyaccelerate.cpu import CPUInfo
        info = CPUInfo(
            brand="Snapdragon 8 Gen 3",
            logical_cores=8,
            physical_cores=8,
            arm_clusters={"Cortex-X4": [0], "Cortex-A720": [1, 2, 3], "Cortex-A520": [4, 5, 6, 7]},
        )
        label = info.short_label()
        assert "Snapdragon" in label
        assert "8C" in label

    def test_check_is_arm_true(self):
        from pyaccelerate.cpu import _check_is_arm
        assert _check_is_arm("aarch64") is True
        assert _check_is_arm("arm64") is True
        assert _check_is_arm("armv7l") is True
        assert _check_is_arm("armv8") is True

    def test_check_is_arm_false(self):
        from pyaccelerate.cpu import _check_is_arm
        assert _check_is_arm("x86_64") is False
        assert _check_is_arm("AMD64") is False
        assert _check_is_arm("i686") is False


# ═══════════════════════════════════════════════════════════════════════════
#  GPU ARM vendor detection
# ═══════════════════════════════════════════════════════════════════════════

class TestGPUARMVendor:
    """Test ARM GPU vendor detection in gpu.detector."""

    def test_adreno_vendor(self):
        from pyaccelerate.gpu.detector import _vendor_from_name
        vendor, discrete = _vendor_from_name("Adreno 750")
        assert vendor == "Qualcomm"
        assert discrete is False

    def test_mali_vendor(self):
        from pyaccelerate.gpu.detector import _vendor_from_name
        vendor, discrete = _vendor_from_name("Mali-G78 MP14")
        assert vendor == "ARM"
        assert discrete is False

    def test_immortalis_vendor(self):
        from pyaccelerate.gpu.detector import _vendor_from_name
        vendor, discrete = _vendor_from_name("Immortalis-G720 MC12")
        assert vendor == "ARM"
        assert discrete is False

    def test_xclipse_vendor(self):
        from pyaccelerate.gpu.detector import _vendor_from_name
        vendor, discrete = _vendor_from_name("Xclipse 920")
        assert vendor == "Samsung"
        assert discrete is False

    def test_powervr_vendor(self):
        from pyaccelerate.gpu.detector import _vendor_from_name
        vendor, discrete = _vendor_from_name("PowerVR BXE-4-32")
        assert vendor == "Imagination"
        assert discrete is False

    def test_maleoon_vendor(self):
        from pyaccelerate.gpu.detector import _vendor_from_name
        vendor, discrete = _vendor_from_name("Maleoon 910")
        assert vendor == "HiSilicon"
        assert discrete is False

    def test_gpu_names_match_same(self):
        from pyaccelerate.gpu.detector import _gpu_names_match
        assert _gpu_names_match("Adreno 750", "Qualcomm Adreno 750") is True

    def test_gpu_names_match_different(self):
        from pyaccelerate.gpu.detector import _gpu_names_match
        assert _gpu_names_match("Adreno 750", "Mali-G78") is False

    def test_gpu_names_match_mali_variants(self):
        from pyaccelerate.gpu.detector import _gpu_names_match
        assert _gpu_names_match("Mali-G78 MP14", "ARM Mali-G78") is True


# ═══════════════════════════════════════════════════════════════════════════
#  NPU ARM vendor detection
# ═══════════════════════════════════════════════════════════════════════════

class TestNPUARMVendor:
    """Test ARM NPU vendor detection in npu.detector."""

    def test_vendor_mediatek(self):
        from pyaccelerate.npu.detector import _vendor_from_name
        assert _vendor_from_name("MediaTek APU 790") == "MediaTek"
        assert _vendor_from_name("Dimensity 9300") == "MediaTek"

    def test_vendor_samsung(self):
        from pyaccelerate.npu.detector import _vendor_from_name
        assert _vendor_from_name("Samsung NPU") == "Samsung"
        assert _vendor_from_name("Exynos 2200") == "Samsung"

    def test_vendor_google(self):
        from pyaccelerate.npu.detector import _vendor_from_name
        assert _vendor_from_name("Google TPU v3") == "Google"
        assert _vendor_from_name("Tensor G4") == "Google"

    def test_vendor_hisilicon(self):
        from pyaccelerate.npu.detector import _vendor_from_name
        assert _vendor_from_name("Da Vinci NPU") == "HiSilicon"
        assert _vendor_from_name("Kirin 9000") == "HiSilicon"

    def test_tops_snapdragon_mobile(self):
        from pyaccelerate.npu.detector import _estimate_tops
        assert _estimate_tops("Snapdragon 8 Gen 3") == 73.0
        assert _estimate_tops("Snapdragon 8 Elite") == 80.0
        assert _estimate_tops("Snapdragon 888") == 26.0

    def test_tops_exynos(self):
        from pyaccelerate.npu.detector import _estimate_tops
        assert _estimate_tops("Exynos 2200") == 34.7
        assert _estimate_tops("Exynos 2500") == 50.0

    def test_tops_tensor(self):
        from pyaccelerate.npu.detector import _estimate_tops
        assert _estimate_tops("Tensor G3") == 30.0
        assert _estimate_tops("Tensor G4") == 35.0

    def test_tops_dimensity(self):
        from pyaccelerate.npu.detector import _estimate_tops
        assert _estimate_tops("Dimensity 9300") == 46.0
        assert _estimate_tops("Dimensity 9200") == 35.0

    def test_tops_kirin(self):
        from pyaccelerate.npu.detector import _estimate_tops
        assert _estimate_tops("Kirin 9000") == 12.0
        assert _estimate_tops("Kirin 9010") == 25.0

    def test_install_hint_arm_npu(self):
        from pyaccelerate.npu.detector import NPUDevice, get_install_hint, reset_cache, _all_npus, _detected
        # Create a mock scenario with an ARM NPU detected but not usable
        import pyaccelerate.npu.detector as mod
        old_all = mod._all_npus
        old_det = mod._detected
        old_best = mod._best_npu
        try:
            mod._all_npus = [NPUDevice(name="Samsung NPU", vendor="Samsung", backend="none", tops=26.0)]
            mod._detected = True
            mod._best_npu = mod._all_npus[0]
            hint = get_install_hint()
            assert "tflite" in hint.lower()
        finally:
            mod._all_npus = old_all
            mod._detected = old_det
            mod._best_npu = old_best


# ═══════════════════════════════════════════════════════════════════════════
#  Engine/CLI ARM integration
# ═══════════════════════════════════════════════════════════════════════════

class TestEngineARMFields:
    """Test that engine.as_dict includes ARM fields."""

    def test_as_dict_has_arm_fields(self):
        from pyaccelerate.engine import Engine
        engine = Engine()
        d = engine.as_dict()
        assert "is_arm" in d["cpu"]
        assert "arm_clusters" in d["cpu"]
        assert "soc_name" in d["cpu"]
        assert "is_android" in d["cpu"]


class TestCLIAndroid:
    """Test CLI android subcommand."""

    def test_cli_android_not_arm(self):
        from pyaccelerate.cli import main
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with patch("platform.machine", return_value="x86_64"):
            with redirect_stdout(f):
                main(["android"])
        assert "Not running on ARM" in f.getvalue()

