"""Tests for pyaccelerate.max_mode module."""

import math
import time
from unittest.mock import patch, MagicMock

import pytest

from pyaccelerate.max_mode import (
    MaxMode,
    MaxModeState,
    activate_max_mode,
    deactivate_max_mode,
    _global_buffer_pool,
)
from pyaccelerate.priority import TaskPriority, EnergyProfile, get_task_priority


# ═══════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════

def _identity(x):
    return x


def _add(a, b):
    return a + b


def _cpu_work(n):
    return sum(math.sqrt(i) for i in range(n))


def _io_sim(ms):
    time.sleep(ms / 1000)
    return ms


def _failing():
    raise ValueError("intentional failure")


# ═══════════════════════════════════════════════════════════════════════
#  MaxModeState
# ═══════════════════════════════════════════════════════════════════════

class TestMaxModeState:
    def test_defaults(self):
        s = MaxModeState()
        assert s.previous_priority == TaskPriority.NORMAL
        assert s.previous_energy == EnergyProfile.BALANCED
        assert s.active is False
        assert s.activation_time == 0.0
        assert s.hardware_manifest == {}

    def test_is_dataclass(self):
        s = MaxModeState()
        assert hasattr(s, "__dataclass_fields__")

    def test_mutable(self):
        s = MaxModeState()
        s.active = True
        s.activation_time = 1.0
        assert s.active is True


# ═══════════════════════════════════════════════════════════════════════
#  activate / deactivate
# ═══════════════════════════════════════════════════════════════════════

class TestActivateDeactivate:
    def test_activate_returns_state(self):
        state = activate_max_mode(set_priority=True, set_energy=False)
        assert isinstance(state, MaxModeState)
        assert state.active is True
        deactivate_max_mode(state)

    def test_deactivate_returns_summary(self):
        state = activate_max_mode(set_priority=True, set_energy=False)
        summary = deactivate_max_mode(state)
        assert isinstance(summary, dict)
        assert "elapsed_s" in summary
        assert "restored_priority" in summary

    def test_deactivate_inactive_returns_error(self):
        state = MaxModeState()
        result = deactivate_max_mode(state)
        assert "error" in result

    def test_hardware_manifest_populated(self):
        state = activate_max_mode(set_priority=True, set_energy=False)
        m = state.hardware_manifest
        assert "cpu" in m
        assert "cpu_physical_cores" in m
        assert "cpu_logical_cores" in m
        assert "gpu_count" in m
        assert "npu_count" in m
        assert "ram_total_gb" in m
        assert "io_pool_size" in m
        assert "cpu_pool_size" in m
        assert m["cpu_physical_cores"] >= 1
        assert m["cpu_logical_cores"] >= 1
        deactivate_max_mode(state)

    def test_activates_high_priority(self):
        state = activate_max_mode(set_priority=True, set_energy=False)
        try:
            p = get_task_priority()
            assert p == TaskPriority.HIGH
        finally:
            deactivate_max_mode(state)

    def test_restores_previous_priority(self):
        from pyaccelerate.priority import set_task_priority
        set_task_priority(TaskPriority.NORMAL)
        state = activate_max_mode(set_priority=True, set_energy=False)
        deactivate_max_mode(state)
        assert get_task_priority() == TaskPriority.NORMAL

    def test_no_priority_change_when_disabled(self):
        from pyaccelerate.priority import set_task_priority
        set_task_priority(TaskPriority.NORMAL)
        state = activate_max_mode(set_priority=False, set_energy=False)
        p = get_task_priority()
        assert p == TaskPriority.NORMAL
        deactivate_max_mode(state)

    def test_buffer_prefill(self):
        state = activate_max_mode(
            set_priority=False, set_energy=False,
            prefill_buffers=True, max_buffers=8,
        )
        stats = _global_buffer_pool.stats
        assert stats["pooled"] >= 1
        deactivate_max_mode(state)

    def test_elapsed_time(self):
        state = activate_max_mode(set_priority=False, set_energy=False)
        time.sleep(0.05)
        summary = deactivate_max_mode(state)
        assert summary["elapsed_s"] >= 0.04

    def test_double_deactivate_safe(self):
        state = activate_max_mode(set_priority=False, set_energy=False)
        deactivate_max_mode(state)
        result = deactivate_max_mode(state)
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════
#  MaxMode context manager
# ═══════════════════════════════════════════════════════════════════════

class TestMaxModeContextManager:
    def test_enters_and_exits(self):
        with MaxMode(set_priority=True, set_energy=False) as m:
            assert m.active is True
        assert m.active is False

    def test_manifest_property(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            manifest = m.manifest
            assert isinstance(manifest, dict)
            assert "cpu" in manifest

    def test_buffer_pool_property(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            pool = m.buffer_pool
            assert pool is _global_buffer_pool

    def test_summary_active(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            s = m.summary()
            assert "ACTIVE" in s
            assert "CPU" in s

    def test_summary_inactive(self):
        m = MaxMode(set_priority=False, set_energy=False)
        s = m.summary()
        assert "INACTIVE" in s


# ═══════════════════════════════════════════════════════════════════════
#  run_io
# ═══════════════════════════════════════════════════════════════════════

class TestRunIO:
    def test_basic(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_io(
                _identity,
                [(1,), (2,), (3,)],
                show_progress=False,
            )
        assert len(results) == 3
        assert set(results) == {1, 2, 3}

    def test_add(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_io(
                _add,
                [(1, 2), (3, 4), (5, 6)],
                show_progress=False,
            )
        assert set(results) == {3, 7, 11}

    def test_empty_items(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_io(_identity, [], show_progress=False)
        assert results == []

    def test_custom_concurrency(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_io(
                _identity,
                [(i,) for i in range(10)],
                max_concurrent=2,
                show_progress=False,
            )
        assert len(results) == 10


# ═══════════════════════════════════════════════════════════════════════
#  run_cpu
# ═══════════════════════════════════════════════════════════════════════

class TestRunCPU:
    def test_basic(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_cpu(
                _cpu_work,
                [(100,), (200,), (300,)],
            )
        assert len(results) == 3
        # All should be positive floats
        for r in results:
            assert isinstance(r, float)
            assert r > 0

    def test_order_preserved(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_cpu(
                _identity,
                [(1,), (2,), (3,), (4,), (5,)],
            )
        assert results == [1, 2, 3, 4, 5]

    def test_empty_items(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_cpu(_identity, [])
        assert results == []


# ═══════════════════════════════════════════════════════════════════════
#  run_all
# ═══════════════════════════════════════════════════════════════════════

class TestRunAll:
    def test_cpu_and_io(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_all(
                cpu_fn=_identity,
                cpu_items=[(1,), (2,), (3,)],
                io_fn=_identity,
                io_items=[(10,), (20,), (30,)],
            )
        assert "cpu" in results
        assert "io" in results
        assert "elapsed_s" in results
        assert len(results["cpu"]) == 3
        assert len(results["io"]) == 3

    def test_cpu_only(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_all(
                cpu_fn=_identity,
                cpu_items=[(1,), (2,)],
            )
        assert "cpu" in results
        assert "io" not in results

    def test_io_only(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_all(
                io_fn=_identity,
                io_items=[(1,), (2,)],
            )
        assert "io" in results
        assert "cpu" not in results

    def test_elapsed_recorded(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_all(
                io_fn=_identity,
                io_items=[(1,)],
            )
        assert results["elapsed_s"] >= 0

    def test_empty_no_crash(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_all()
        assert "elapsed_s" in results


# ═══════════════════════════════════════════════════════════════════════
#  run_pipeline
# ═══════════════════════════════════════════════════════════════════════

class TestRunPipeline:
    def test_single_stage(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_pipeline([
                ("double", lambda x: x * 2, [(1,), (2,), (3,)]),
            ])
        assert "double" in results
        assert set(results["double"]) == {2, 4, 6}

    def test_multi_stage(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_pipeline([
                ("generate", lambda x: x, [(10,), (20,), (30,)]),
                ("double", lambda x: x * 2, [(5,), (15,), (25,)]),
            ])
        assert "generate" in results
        assert "double" in results
        assert len(results["generate"]) == 3
        assert len(results["double"]) == 3

    def test_empty_pipeline(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_pipeline([])
        assert results == {}

    def test_stage_concurrency_override(self):
        with MaxMode(set_priority=False, set_energy=False) as m:
            results = m.run_pipeline(
                [("task", _identity, [(i,) for i in range(5)])],
                stage_concurrency={"task": 2},
            )
        assert len(results["task"]) == 5


# ═══════════════════════════════════════════════════════════════════════
#  Integration
# ═══════════════════════════════════════════════════════════════════════

class TestMaxModeIntegration:
    def test_full_workflow(self):
        """Activate → run IO → run CPU → deactivate."""
        with MaxMode(set_priority=True, set_energy=False) as m:
            io_res = m.run_io(
                _identity,
                [(i,) for i in range(5)],
                show_progress=False,
            )
            cpu_res = m.run_cpu(
                _cpu_work,
                [(100,), (200,)],
            )
            assert len(io_res) == 5
            assert len(cpu_res) == 2
            s = m.summary()
            assert "ACTIVE" in s

    def test_nested_max_modes(self):
        """Nested context managers should work without crashing."""
        with MaxMode(set_priority=False, set_energy=False) as outer:
            with MaxMode(set_priority=False, set_energy=False) as inner:
                results = inner.run_io(_identity, [(1,)], show_progress=False)
                assert results == [1]
