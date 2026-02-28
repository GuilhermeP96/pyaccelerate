"""Tests for pyaccelerate.priority module."""

import platform
from unittest.mock import patch, MagicMock

import pytest

from pyaccelerate.priority import (
    TaskPriority,
    EnergyProfile,
    set_task_priority,
    get_task_priority,
    set_energy_profile,
    get_energy_profile,
    set_io_priority,
    get_priority_info,
    max_performance,
    balanced,
    power_saver,
    _WIN_POWER_PLANS,
)


# ═══════════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════════

class TestTaskPriority:
    def test_values(self):
        assert TaskPriority.IDLE == 0
        assert TaskPriority.BELOW_NORMAL == 1
        assert TaskPriority.NORMAL == 2
        assert TaskPriority.ABOVE_NORMAL == 3
        assert TaskPriority.HIGH == 4
        assert TaskPriority.REALTIME == 5

    def test_ordering(self):
        assert TaskPriority.IDLE < TaskPriority.NORMAL < TaskPriority.HIGH
        assert TaskPriority.REALTIME > TaskPriority.HIGH

    def test_names(self):
        for p in TaskPriority:
            assert p.name == p.name.upper()
            assert isinstance(p.value, int)

    def test_member_count(self):
        assert len(TaskPriority) == 6


class TestEnergyProfile:
    def test_values(self):
        assert EnergyProfile.POWER_SAVER == 0
        assert EnergyProfile.BALANCED == 1
        assert EnergyProfile.PERFORMANCE == 2
        assert EnergyProfile.ULTRA_PERFORMANCE == 3

    def test_ordering(self):
        assert EnergyProfile.POWER_SAVER < EnergyProfile.BALANCED
        assert EnergyProfile.PERFORMANCE < EnergyProfile.ULTRA_PERFORMANCE

    def test_member_count(self):
        assert len(EnergyProfile) == 4


# ═══════════════════════════════════════════════════════════════════════
#  Task Priority API
# ═══════════════════════════════════════════════════════════════════════

class TestSetTaskPriority:
    def test_set_normal_returns_bool(self):
        result = set_task_priority(TaskPriority.NORMAL)
        assert isinstance(result, bool)

    def test_set_and_get_normal(self):
        set_task_priority(TaskPriority.NORMAL)
        p = get_task_priority()
        assert p == TaskPriority.NORMAL

    def test_set_above_normal(self):
        result = set_task_priority(TaskPriority.ABOVE_NORMAL)
        # May fail on CI without privileges, but should not raise
        assert isinstance(result, bool)
        # Restore
        set_task_priority(TaskPriority.NORMAL)

    def test_get_returns_enum(self):
        p = get_task_priority()
        assert isinstance(p, TaskPriority)

    @patch("pyaccelerate.priority.platform")
    def test_unsupported_os_returns_false(self, mock_platform):
        mock_platform.system.return_value = "FakeOS"
        result = set_task_priority(TaskPriority.HIGH)
        assert result is False

    @patch("pyaccelerate.priority.platform")
    def test_get_unsupported_returns_normal(self, mock_platform):
        mock_platform.system.return_value = "FakeOS"
        result = get_task_priority()
        assert result == TaskPriority.NORMAL


class TestGetTaskPriority:
    def test_returns_task_priority(self):
        p = get_task_priority()
        assert isinstance(p, TaskPriority)


# ═══════════════════════════════════════════════════════════════════════
#  IO Priority
# ═══════════════════════════════════════════════════════════════════════

class TestIOPriority:
    def test_returns_bool(self):
        result = set_io_priority(high=True)
        assert isinstance(result, bool)

    def test_non_linux_returns_false(self):
        if platform.system() != "Linux":
            assert set_io_priority(high=True) is False

    def test_low_priority_returns_bool(self):
        result = set_io_priority(high=False)
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════════════════════
#  Energy Profile API
# ═══════════════════════════════════════════════════════════════════════

class TestSetEnergyProfile:
    def test_set_returns_bool(self):
        result = set_energy_profile(EnergyProfile.BALANCED)
        assert isinstance(result, bool)

    def test_get_returns_enum(self):
        p = get_energy_profile()
        assert isinstance(p, EnergyProfile)

    @patch("pyaccelerate.priority.platform")
    def test_unsupported_os_returns_false(self, mock_platform):
        mock_platform.system.return_value = "FakeOS"
        result = set_energy_profile(EnergyProfile.PERFORMANCE)
        assert result is False

    @patch("pyaccelerate.priority.platform")
    def test_get_unsupported_returns_balanced(self, mock_platform):
        mock_platform.system.return_value = "FakeOS"
        result = get_energy_profile()
        assert result == EnergyProfile.BALANCED


class TestWindowsPowerPlans:
    def test_all_profiles_have_guids(self):
        for profile in EnergyProfile:
            assert profile in _WIN_POWER_PLANS
            guid = _WIN_POWER_PLANS[profile]
            assert isinstance(guid, str)
            # GUID format: 8-4-4-4-12
            parts = guid.split("-")
            assert len(parts) == 5

    @patch("pyaccelerate.priority.platform")
    @patch("pyaccelerate.priority.subprocess")
    def test_set_energy_windows_calls_powercfg(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Windows"
        mock_sub.run.return_value = MagicMock(returncode=0)
        result = set_energy_profile(EnergyProfile.PERFORMANCE)
        assert result is True
        mock_sub.run.assert_called()

    @patch("pyaccelerate.priority.platform")
    @patch("pyaccelerate.priority.subprocess")
    def test_get_energy_windows(self, mock_sub, mock_plat):
        mock_plat.system.return_value = "Windows"
        guid = _WIN_POWER_PLANS[EnergyProfile.BALANCED]
        mock_sub.run.return_value = MagicMock(
            returncode=0,
            stdout=f"Power Scheme GUID: {guid}  (Balanced)"
        )
        result = get_energy_profile()
        assert result == EnergyProfile.BALANCED


# ═══════════════════════════════════════════════════════════════════════
#  Convenience Presets
# ═══════════════════════════════════════════════════════════════════════

class TestPresets:
    def test_max_performance_returns_dict(self):
        result = max_performance()
        assert isinstance(result, dict)
        assert "task_priority" in result
        assert "energy_profile" in result
        assert isinstance(result["task_priority"], bool)
        # Restore
        balanced()

    def test_balanced_returns_dict(self):
        result = balanced()
        assert isinstance(result, dict)
        assert "task_priority" in result

    def test_power_saver_returns_dict(self):
        result = power_saver()
        assert isinstance(result, dict)
        assert "task_priority" in result
        assert "energy_profile" in result
        assert "io_priority" in result
        # Restore
        balanced()

    def test_max_performance_sets_high_priority(self):
        max_performance()
        p = get_task_priority()
        assert p == TaskPriority.HIGH
        # Restore
        balanced()

    def test_balanced_sets_normal_priority(self):
        balanced()
        p = get_task_priority()
        assert p == TaskPriority.NORMAL

    def test_power_saver_sets_below_normal_priority(self):
        power_saver()
        p = get_task_priority()
        assert p == TaskPriority.BELOW_NORMAL
        # Restore
        balanced()


# ═══════════════════════════════════════════════════════════════════════
#  Info
# ═══════════════════════════════════════════════════════════════════════

class TestPriorityInfo:
    def test_returns_dict(self):
        info = get_priority_info()
        assert isinstance(info, dict)

    def test_keys(self):
        info = get_priority_info()
        assert "task_priority" in info
        assert "energy_profile" in info
        assert "os" in info

    def test_values_are_strings(self):
        info = get_priority_info()
        for v in info.values():
            assert isinstance(v, str)

    def test_os_matches_platform(self):
        info = get_priority_info()
        assert info["os"] == platform.system()

    def test_priority_is_valid_name(self):
        info = get_priority_info()
        valid = {p.name for p in TaskPriority}
        assert info["task_priority"] in valid

    def test_energy_is_valid_name(self):
        info = get_priority_info()
        valid = {p.name for p in EnergyProfile}
        assert info["energy_profile"] in valid
