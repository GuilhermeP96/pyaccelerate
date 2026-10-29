"""Report regressions: closed borders, terminal cells and complete content."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pyaccelerate._report import render_report, visual_width
from pyaccelerate.autotune import TuneProfile, profile_summary
from pyaccelerate.engine import Engine
from pyaccelerate.max_mode import MaxMode, MaxModeState


def assert_box(report, limit):
    lines = report.splitlines()
    widths = {visual_width(line) for line in lines}
    assert len(widths) == 1
    assert widths.pop() <= limit
    assert lines[0].startswith("╔") and lines[0].endswith("╗")
    assert lines[-1].startswith("╚") and lines[-1].endswith("╝")
    assert all(line[0] in "║╠" and line[-1] in "║╣" for line in lines[1:-1])


@pytest.mark.parametrize("width", [6, 20, 40, 80, 120])
def test_wrap_preserves_content_and_closes_every_line(width):
    value = "abcdefghijklmnopqrstuvwxyz" * 7
    report = render_report(
        "Report", ["CPU: " + value, "GPU: 漢字 café e\u0301", None, "RAM: 8 GB"], width=width
    )
    assert_box(report, width)
    joined = "".join(line[2:-2].strip() for line in report.splitlines() if line.startswith("║"))
    assert value in joined
    assert "漢字" in joined


def test_content_width_and_aligned_labels():
    report = render_report("HW", ["CPU: A", "Memory: B"], width=80)
    assert visual_width(report.splitlines()[0]) == len("Memory: B") + 4
    lines = [line for line in report.splitlines() if line.startswith("║")][1:]
    assert lines[0].index("A") == lines[1].index("B")


def test_ansi_tabs_combining_wide_and_joined_emoji():
    report = render_report(
        "Color", ["CPU: \x1b[31m漢字 e\u0301 👩\u200d💻\x1b[0m " * 5, "GPU:\tNone"], width=24
    )
    assert_box(report, 24)
    assert "\t" not in report
    assert "\x1b[31m" in report
    assert "👩\u200d💻" in report
    assert visual_width("\x1b[31m漢字 e\u0301 👩\u200d💻\x1b[0m") == 9


@pytest.mark.parametrize("width", [24, 80, 140])
def test_all_reports_respect_terminal_width(width):
    with patch(
        "pyaccelerate._report.shutil.get_terminal_size",
        return_value=SimpleNamespace(columns=width),
    ):
        with (
            patch("pyaccelerate.engine.detect_gpus", return_value=[]),
            patch("pyaccelerate.engine.detect_npus", return_value=[]),
        ):
            engine = Engine()
        engine._cpu = replace(
            engine._cpu,
            brand="AMD EPYC 9114 96-Core Processor " * 5,
            flags=tuple("avx512verylongflag" + str(i) for i in range(8)),
        )
        report = engine.summary()
        assert_box(report, width)
        assert "avx512" in report
        engine.shutdown()
        mode = MaxMode(set_priority=False, set_energy=False)
        mode._state = MaxModeState(active=True, hardware_manifest={"cpu": "CPU model " * 30})
        assert_box(mode.summary(), width)
        assert_box(profile_summary(TuneProfile(gpu_backend="backend " * 25)), width)


def test_multiline_values_and_terminal_fallback(monkeypatch):
    monkeypatch.setenv("COLUMNS", "40")
    report = render_report("Title", ["CPU: line one\nline two\tend"])
    assert_box(report, 40)
    assert "line one" in report and "line two" in report
