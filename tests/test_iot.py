"""Tests for pyaccelerate.iot — IoT / SBC platform detection."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from pyaccelerate.iot import (
    SBCInfo,
    _SBCSoC,
    _SBC_SOC_DATABASE,
    _BOARD_FAMILIES,
    _detect_family,
    _detect_gpio,
    _detect_fan,
    _detect_ram_mb,
    _detect_storage_type,
    _lookup_sbc_soc,
    _read_dt_compatible,
    _read_dt_model,
    detect_coral_tpu,
    detect_sbc,
    get_jetson_power_modes,
    get_sbc_summary,
    get_sbc_thermal,
    is_circuitpython,
    is_jetson,
    is_micropython,
    is_raspberry_pi,
    is_sbc,
    recommend_iot_workers,
    reset_cache,
    set_jetson_power_mode,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear all caches before each test."""
    reset_cache()
    yield
    reset_cache()


# ═══════════════════════════════════════════════════════════════════════════
#  SoC Database Integrity
# ═══════════════════════════════════════════════════════════════════════════

class TestSoCDatabase:
    """Verify the SBC SoC database has valid entries."""

    def test_database_not_empty(self):
        assert len(_SBC_SOC_DATABASE) > 0

    def test_all_entries_have_soc_name(self):
        for key, soc in _SBC_SOC_DATABASE.items():
            assert soc.soc_name, f"Missing soc_name for key '{key}'"

    def test_all_entries_have_vendor(self):
        for key, soc in _SBC_SOC_DATABASE.items():
            assert soc.vendor, f"Missing vendor for key '{key}'"

    def test_all_entries_have_cpu_arch(self):
        for key, soc in _SBC_SOC_DATABASE.items():
            assert soc.cpu_arch, f"Missing cpu_arch for key '{key}'"

    def test_all_entries_have_positive_cores(self):
        for key, soc in _SBC_SOC_DATABASE.items():
            assert soc.cpu_cores > 0, f"Zero cores for key '{key}'"

    def test_all_entries_have_positive_cpu_mhz(self):
        for key, soc in _SBC_SOC_DATABASE.items():
            assert soc.cpu_max_mhz > 0, f"Zero frequency for key '{key}'"

    def test_raspberry_pi_entries(self):
        rpi_keys = [k for k in _SBC_SOC_DATABASE if k.startswith("bcm")]
        assert len(rpi_keys) >= 4  # BCM2712, 2711, 2837, 2836, 2835

    def test_jetson_entries(self):
        jetson_keys = [k for k in _SBC_SOC_DATABASE if "jetson" in k]
        assert len(jetson_keys) >= 6  # Orin Nano, NX, AGX, Xavier NX, AGX Xavier, TX2, Nano

    def test_jetson_cuda_cores(self):
        for key, soc in _SBC_SOC_DATABASE.items():
            if "jetson" in key:
                assert soc.gpu_cuda_cores > 0, f"Jetson '{key}' should have CUDA cores"

    def test_allwinner_entries(self):
        aw_keys = [k for k in _SBC_SOC_DATABASE if k.startswith("h") and k[1:].isdigit()
                    or k == "a64" or k.startswith("h6")]
        assert len(aw_keys) >= 4

    def test_rockchip_entries(self):
        rk_keys = [k for k in _SBC_SOC_DATABASE if k.startswith("rk")]
        assert len(rk_keys) >= 4

    def test_nxp_entries(self):
        nxp_keys = [k for k in _SBC_SOC_DATABASE if k.startswith("imx")]
        assert len(nxp_keys) >= 2

    def test_board_families_not_empty(self):
        assert len(_BOARD_FAMILIES) >= 8


# ═══════════════════════════════════════════════════════════════════════════
#  SoC Lookup
# ═══════════════════════════════════════════════════════════════════════════

class TestSoCLookup:
    """Test _lookup_sbc_soc matching."""

    def test_exact_key_match(self):
        soc = _lookup_sbc_soc("bcm2711", "")
        assert soc is not None
        assert soc.soc_name == "BCM2711"

    def test_compatible_substring_match(self):
        soc = _lookup_sbc_soc("", "brcm,bcm2711")
        assert soc is not None
        assert soc.soc_name == "BCM2711"

    def test_jetson_model_match(self):
        soc = _lookup_sbc_soc("Jetson AGX Orin Developer Kit", "")
        assert soc is not None
        assert "Orin" in soc.soc_name

    def test_rockchip_match(self):
        soc = _lookup_sbc_soc("Rockchip RK3588 board", "")
        assert soc is not None
        assert soc.soc_name == "RK3588"

    def test_no_match_returns_none(self):
        soc = _lookup_sbc_soc("totally unknown board xyz", "")
        assert soc is None

    def test_empty_strings_no_crash(self):
        soc = _lookup_sbc_soc("", "")
        assert soc is None


# ═══════════════════════════════════════════════════════════════════════════
#  Family Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestFamilyDetection:
    def test_raspberry_pi_family(self):
        assert _detect_family("Raspberry Pi 4 Model B", "") == "raspberry_pi"

    def test_jetson_family(self):
        assert _detect_family("NVIDIA Jetson Orin Nano", "") == "jetson"

    def test_beaglebone_family(self):
        assert _detect_family("BeagleBone Black", "") == "beaglebone"

    def test_orange_pi_family(self):
        assert _detect_family("Orange Pi 5", "") == "orange_pi"

    def test_pine64_family(self):
        assert _detect_family("Pine64 Plus", "") == "pine64"

    def test_coral_family(self):
        assert _detect_family("Google Coral Dev Board", "") == "coral"

    def test_unknown_family(self):
        assert _detect_family("Unknown Board XYZ", "") == "generic"


# ═══════════════════════════════════════════════════════════════════════════
#  MicroPython / CircuitPython Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestRuntimeDetection:
    def test_micropython_false_on_cpython(self):
        assert is_micropython() is False

    def test_circuitpython_false_on_cpython(self):
        assert is_circuitpython() is False

    def test_micropython_true_when_mocked(self):
        reset_cache()
        impl = MagicMock()
        impl.name = "micropython"
        with patch.object(sys, "implementation", impl):
            reset_cache()
            assert is_micropython() is True

    def test_circuitpython_true_when_mocked(self):
        reset_cache()
        impl = MagicMock()
        impl.name = "circuitpython"
        with patch.object(sys, "implementation", impl):
            reset_cache()
            assert is_circuitpython() is True


# ═══════════════════════════════════════════════════════════════════════════
#  is_sbc Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestIsSBC:
    def test_not_sbc_on_desktop(self):
        """Standard x86_64 desktop should not be detected as SBC."""
        with patch("pyaccelerate.iot._read_dt_model", return_value=""), \
             patch("pyaccelerate.iot._read_dt_compatible", return_value=""), \
             patch("pyaccelerate.iot.is_micropython", return_value=False), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=False), \
             patch("os.path.isfile", return_value=False):
            reset_cache()
            assert is_sbc() is False

    def test_raspberry_pi_detected(self):
        with patch("pyaccelerate.iot._read_dt_model", return_value="Raspberry Pi 4 Model B"), \
             patch("pyaccelerate.iot._read_dt_compatible", return_value="brcm,bcm2711"), \
             patch("pyaccelerate.iot.is_micropython", return_value=False), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=False):
            reset_cache()
            assert is_sbc() is True

    def test_jetson_detected_via_model(self):
        with patch("pyaccelerate.iot._read_dt_model", return_value="NVIDIA Jetson AGX Orin"), \
             patch("pyaccelerate.iot._read_dt_compatible", return_value="nvidia,tegra234"), \
             patch("pyaccelerate.iot.is_micropython", return_value=False), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=False):
            reset_cache()
            assert is_sbc() is True

    def test_jetson_detected_via_tegra_file(self):
        with patch("pyaccelerate.iot._read_dt_model", return_value=""), \
             patch("pyaccelerate.iot._read_dt_compatible", return_value=""), \
             patch("pyaccelerate.iot.is_micropython", return_value=False), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=False), \
             patch("os.path.isfile", return_value=True):
            reset_cache()
            assert is_sbc() is True

    def test_beaglebone_detected(self):
        with patch("pyaccelerate.iot._read_dt_model", return_value="BeagleBone Black"), \
             patch("pyaccelerate.iot._read_dt_compatible", return_value="ti,am3358"), \
             patch("pyaccelerate.iot.is_micropython", return_value=False), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=False):
            reset_cache()
            assert is_sbc() is True

    def test_micropython_is_sbc(self):
        with patch("pyaccelerate.iot.is_micropython", return_value=True), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=False):
            reset_cache()
            assert is_sbc() is True


# ═══════════════════════════════════════════════════════════════════════════
#  is_jetson / is_raspberry_pi
# ═══════════════════════════════════════════════════════════════════════════

class TestBoardSpecificDetection:
    def test_is_jetson_true(self):
        with patch("pyaccelerate.iot._read_dt_model", return_value="NVIDIA Jetson Orin Nano"), \
             patch("pyaccelerate.iot._read_dt_compatible", return_value="nvidia,tegra234"):
            assert is_jetson() is True

    def test_is_jetson_false(self):
        with patch("pyaccelerate.iot._read_dt_model", return_value="Raspberry Pi 4"), \
             patch("pyaccelerate.iot._read_dt_compatible", return_value="brcm,bcm2711"), \
             patch("os.path.isfile", return_value=False):
            assert is_jetson() is False

    def test_is_raspberry_pi_true(self):
        with patch("pyaccelerate.iot._read_dt_model", return_value="Raspberry Pi 5 Model B"):
            assert is_raspberry_pi() is True

    def test_is_raspberry_pi_false(self):
        with patch("pyaccelerate.iot._read_dt_model", return_value="NVIDIA Jetson"):
            assert is_raspberry_pi() is False


# ═══════════════════════════════════════════════════════════════════════════
#  detect_sbc
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectSBC:
    def test_detect_sbc_returns_none_when_not_sbc(self):
        with patch("pyaccelerate.iot.is_sbc", return_value=False):
            reset_cache()
            assert detect_sbc() is None

    def test_detect_sbc_raspberry_pi_4(self):
        with patch("pyaccelerate.iot.is_sbc", return_value=True), \
             patch("pyaccelerate.iot._read_dt_model", return_value="Raspberry Pi 4 Model B Rev 1.4"), \
             patch("pyaccelerate.iot._read_dt_compatible", return_value="brcm,bcm2711"), \
             patch("pyaccelerate.iot._detect_ram_mb", return_value=4096), \
             patch("pyaccelerate.iot._detect_gpio", return_value=(True, 40)), \
             patch("pyaccelerate.iot._detect_peripherals"), \
             patch("pyaccelerate.iot._detect_fan", return_value=False), \
             patch("pyaccelerate.iot.is_jetson", return_value=False):
            reset_cache()
            sbc = detect_sbc()
            assert sbc is not None
            assert sbc.board_name == "Raspberry Pi 4 Model B Rev 1.4"
            assert sbc.soc_name == "BCM2711"
            assert sbc.soc_vendor == "Broadcom"
            assert sbc.gpu_name == "VideoCore VI"
            assert sbc.family == "raspberry_pi"
            assert sbc.ram_mb == 4096
            assert sbc.cpu_cores == 4
            assert sbc.has_gpio is True
            assert sbc.gpio_pins == 40

    def test_detect_sbc_jetson_orin(self):
        with patch("pyaccelerate.iot.is_sbc", return_value=True), \
             patch("pyaccelerate.iot._read_dt_model", return_value="NVIDIA Jetson AGX Orin Developer Kit"), \
             patch("pyaccelerate.iot._read_dt_compatible", return_value="nvidia,tegra234"), \
             patch("pyaccelerate.iot._detect_ram_mb", return_value=32768), \
             patch("pyaccelerate.iot._detect_gpio", return_value=(True, 40)), \
             patch("pyaccelerate.iot._detect_peripherals"), \
             patch("pyaccelerate.iot._detect_fan", return_value=True), \
             patch("pyaccelerate.iot.is_jetson", return_value=True), \
             patch("pyaccelerate.iot._detect_jetson_extras"):
            reset_cache()
            sbc = detect_sbc()
            assert sbc is not None
            assert "Jetson" in sbc.board_name
            assert "Orin" in sbc.soc_name
            assert sbc.gpu_cuda_cores == 2048
            assert sbc.npu_tops == 275.0
            assert sbc.family == "jetson"

    def test_detect_sbc_cached(self):
        """Second call should return cached result."""
        with patch("pyaccelerate.iot.is_sbc", return_value=True), \
             patch("pyaccelerate.iot._read_dt_model", return_value="Raspberry Pi 5"), \
             patch("pyaccelerate.iot._read_dt_compatible", return_value="brcm,bcm2712"), \
             patch("pyaccelerate.iot._detect_ram_mb", return_value=8192), \
             patch("pyaccelerate.iot._detect_gpio", return_value=(True, 40)), \
             patch("pyaccelerate.iot._detect_peripherals"), \
             patch("pyaccelerate.iot._detect_fan", return_value=False), \
             patch("pyaccelerate.iot.is_jetson", return_value=False):
            reset_cache()
            sbc1 = detect_sbc()
            sbc2 = detect_sbc()
            assert sbc1 is sbc2  # Same object (cached)


# ═══════════════════════════════════════════════════════════════════════════
#  Worker Recommendations
# ═══════════════════════════════════════════════════════════════════════════

class TestRecommendIoTWorkers:
    def test_micropython_returns_1(self):
        with patch("pyaccelerate.iot.is_micropython", return_value=True), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=False):
            assert recommend_iot_workers() == 1

    def test_circuitpython_returns_1(self):
        with patch("pyaccelerate.iot.is_micropython", return_value=False), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=True):
            assert recommend_iot_workers() == 1

    def test_low_ram_128mb(self):
        with patch("pyaccelerate.iot.is_micropython", return_value=False), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=False), \
             patch("pyaccelerate.iot._detect_ram_mb", return_value=128):
            assert recommend_iot_workers() == 1

    def test_low_ram_512mb(self):
        with patch("pyaccelerate.iot.is_micropython", return_value=False), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=False), \
             patch("pyaccelerate.iot._detect_ram_mb", return_value=400), \
             patch("os.cpu_count", return_value=4):
            assert recommend_iot_workers() == 2

    def test_ram_1gb_cpu_bound(self):
        with patch("pyaccelerate.iot.is_micropython", return_value=False), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=False), \
             patch("pyaccelerate.iot._detect_ram_mb", return_value=900), \
             patch("os.cpu_count", return_value=4):
            assert recommend_iot_workers(io_bound=False) <= 4

    def test_ram_2gb_io_bound(self):
        with patch("pyaccelerate.iot.is_micropython", return_value=False), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=False), \
             patch("pyaccelerate.iot._detect_ram_mb", return_value=1500), \
             patch("os.cpu_count", return_value=4):
            result = recommend_iot_workers(io_bound=True)
            assert result <= 8

    def test_high_ram_standard(self):
        with patch("pyaccelerate.iot.is_micropython", return_value=False), \
             patch("pyaccelerate.iot.is_circuitpython", return_value=False), \
             patch("pyaccelerate.iot._detect_ram_mb", return_value=8192), \
             patch("os.cpu_count", return_value=8):
            result = recommend_iot_workers(io_bound=False)
            assert result == 8


# ═══════════════════════════════════════════════════════════════════════════
#  Thermal
# ═══════════════════════════════════════════════════════════════════════════

class TestSBCThermal:
    def test_thermal_normal(self):
        with patch("pyaccelerate.android.get_thermal_zones",
                    return_value={"cpu-thermal": 45.0}):
            result = get_sbc_thermal()
            assert result["recommendation"] == "normal"
            assert result["is_throttled"] is False

    def test_thermal_warm(self):
        with patch("pyaccelerate.android.get_thermal_zones",
                    return_value={"cpu-thermal": 72.0}):
            result = get_sbc_thermal()
            assert result["cpu_temp_c"] == 72.0
            assert "warm" in result["recommendation"]

    def test_thermal_critical(self):
        with patch("pyaccelerate.android.get_thermal_zones",
                    return_value={"soc-thermal": 85.0}):
            result = get_sbc_thermal()
            assert result["is_throttled"] is True
            assert "critical" in result["recommendation"]


# ═══════════════════════════════════════════════════════════════════════════
#  Coral Edge TPU
# ═══════════════════════════════════════════════════════════════════════════

class TestCoralTPU:
    def test_no_coral_when_lsusb_fails(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert detect_coral_tpu() is None

    def test_coral_usb_detected(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Bus 001 Device 003: ID 18d1:089a Google Inc. Coral USB Accelerator"

        with patch("subprocess.run", return_value=mock_result), \
             patch("pyaccelerate.iot._check_edgetpu_runtime", return_value=True):
            coral = detect_coral_tpu()
            assert coral is not None
            assert coral["type"] == "usb"
            assert "Coral" in coral["name"]


# ═══════════════════════════════════════════════════════════════════════════
#  Jetson Power Modes
# ═══════════════════════════════════════════════════════════════════════════

class TestJetsonPowerModes:
    def test_no_modes_when_not_jetson(self):
        with patch("pyaccelerate.iot.is_jetson", return_value=False):
            assert get_jetson_power_modes() == []

    def test_set_power_mode_when_not_jetson(self):
        with patch("pyaccelerate.iot.is_jetson", return_value=False):
            assert set_jetson_power_mode(0) is False


# ═══════════════════════════════════════════════════════════════════════════
#  get_sbc_summary
# ═══════════════════════════════════════════════════════════════════════════

class TestSBCSummary:
    def test_summary_not_sbc(self):
        with patch("pyaccelerate.iot.detect_sbc", return_value=None):
            result = get_sbc_summary()
            assert result["is_sbc"] is False

    def test_summary_with_sbc(self):
        sbc = SBCInfo(
            board_name="Test Board",
            soc_name="TestSoC",
            soc_vendor="TestVendor",
            cpu_arch="ARMv8-A",
            cpu_cores=4,
            cpu_max_mhz=1500.0,
            ram_mb=2048,
            gpu_name="Test GPU",
            has_gpio=True,
            gpio_pins=40,
            family="test",
        )
        with patch("pyaccelerate.iot.detect_sbc", return_value=sbc), \
             patch("pyaccelerate.iot.recommend_iot_workers", return_value=4):
            result = get_sbc_summary()
            assert result["is_sbc"] is True
            assert result["board_name"] == "Test Board"
            assert result["cpu_cores"] == 4
            assert result["recommended_workers"] == 4


# ═══════════════════════════════════════════════════════════════════════════
#  SBCInfo Dataclass
# ═══════════════════════════════════════════════════════════════════════════

class TestSBCInfo:
    def test_defaults(self):
        info = SBCInfo()
        assert info.board_name == ""
        assert info.ram_mb == 0
        assert info.has_gpio is False
        assert info.gpu_cuda_cores == 0
        assert info.npu_tops == 0.0
        assert info.family == ""

    def test_all_fields(self):
        info = SBCInfo(
            board_name="RPi 5",
            soc_name="BCM2712",
            soc_vendor="Broadcom",
            cpu_arch="ARMv8.2-A",
            cpu_cores=4,
            cpu_max_mhz=2400.0,
            ram_mb=8192,
            gpu_name="VideoCore VII",
            has_gpio=True,
            gpio_pins=40,
            has_pcie=True,
            has_wifi=True,
            has_bluetooth=True,
            has_ethernet=True,
            usb_ports=4,
            storage_type="microSD",
            family="raspberry_pi",
        )
        assert info.soc_name == "BCM2712"
        assert info.cpu_max_mhz == 2400.0
        assert info.has_pcie is True


# ═══════════════════════════════════════════════════════════════════════════
#  Storage Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestStorageDetection:
    def test_no_storage_detected(self):
        with patch("pathlib.Path.exists", return_value=False):
            result = _detect_storage_type()
            # May return empty string
            assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════
#  Fan Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestFanDetection:
    def test_no_fan_on_desktop(self):
        # On typical test environments, no fan sysfs
        with patch("pathlib.Path.exists", return_value=False):
            assert _detect_fan() is False


# ═══════════════════════════════════════════════════════════════════════════
#  Reset Cache
# ═══════════════════════════════════════════════════════════════════════════

class TestResetCache:
    def test_reset_clears_all(self):
        """reset_cache should allow fresh detection."""
        import pyaccelerate.iot as iot_mod
        iot_mod._is_sbc = True
        iot_mod._sbc_info = SBCInfo(board_name="cached")
        reset_cache()
        assert iot_mod._is_sbc is None
        assert iot_mod._sbc_info is None
        assert iot_mod._dt_model_cache is None
