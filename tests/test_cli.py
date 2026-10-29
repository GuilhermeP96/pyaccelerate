"""Tests for machine-aware CLI dependency installation."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pyaccelerate import cli


def _engine(
    *, gpu: bool = False, usable_gpu: bool = False, npu: bool = False, usable_npu: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        gpus=[object()] if gpu else [],
        usable_gpus=[object()] if usable_gpu else [],
        npus=[object()] if npu else [],
        usable_npus=[object()] if usable_npu else [],
    )


def test_missing_dependencies_include_numpy_and_machine_backends(monkeypatch) -> None:
    available = {"psutil"}
    monkeypatch.setattr(cli, "_distribution_available", lambda package: package in available)
    monkeypatch.setattr(
        "pyaccelerate.gpu.get_install_hint",
        lambda: "Install GPU support: pip install cupy-cuda12x",
    )
    monkeypatch.setattr(
        "pyaccelerate.npu.get_install_hint",
        lambda: "Install NPU support: pip install openvino or pip install onnxruntime-openvino",
    )

    packages = cli._missing_dependency_packages(_engine(gpu=True, npu=True))

    assert packages == [
        "numpy>=1.26",
        "cupy-cuda12x",
        "openvino",
        "onnxruntime-openvino",
    ]


def test_installed_backends_are_not_recommended(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_distribution_available", lambda _package: True)
    monkeypatch.setattr(
        "pyaccelerate.gpu.get_install_hint",
        lambda: "Install GPU support: pip install pyopencl",
    )

    assert cli._missing_dependency_packages(_engine(gpu=True)) == []


def test_install_all_uses_active_interpreter_without_prompt(monkeypatch) -> None:
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(cli, "_missing_dependency_packages", lambda _engine: ["numpy>=1.26"])
    monkeypatch.setattr(cli.subprocess, "run", run)

    cli._offer_missing_deps(_engine(), install_all=True)

    run.assert_called_once_with(
        [cli.sys.executable, "-m", "pip", "install", "numpy>=1.26"],
        timeout=900,
        check=False,
    )


def test_noninteractive_info_reports_repair_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_missing_dependency_packages", lambda _engine: ["numpy>=1.26"])
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)

    cli._offer_missing_deps(_engine())

    output = capsys.readouterr().out
    assert "numpy>=1.26" in output
    assert "pyaccelerate info --install-deps" in output


@pytest.mark.parametrize(
    "version,missing", [("1.25.9", True), ("1.26.0", False), ("2.0.0", False)]
)
def test_requirements_check_minimum_versions(monkeypatch, version, missing):
    monkeypatch.setattr(cli.metadata, "version", lambda _name: version)
    assert cli._requirement_missing("numpy>=1.26") is missing


def test_requirements_respect_environment_markers(monkeypatch):
    monkeypatch.setattr(cli, "_distribution_available", lambda _name: False)
    assert not cli._requirement_missing('missing; python_version < "3.0"')
    assert cli._requirement_missing("missing>=1.0")


def test_failed_dependency_install_exits_nonzero(monkeypatch):
    monkeypatch.setattr(cli, "_missing_dependency_packages", lambda _engine: ["numpy>=1.26"])
    monkeypatch.setattr(cli, "_install_packages", lambda _packages: False)
    with pytest.raises(SystemExit) as exc:
        cli._offer_missing_deps(_engine(), install_all=True)
    assert exc.value.code == 1


def test_benchmark_json_stdout_is_parseable(monkeypatch, capsys):
    monkeypatch.setattr("pyaccelerate.benchmark.run_all", lambda **_kwargs: {"cpu": {"score": 1}})
    cli.main(["benchmark", "--json"])
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"cpu": {"score": 1}}
    assert "Running benchmarks" in captured.err


def test_tune_json_apply_stdout_is_parseable(monkeypatch, capsys):
    from pyaccelerate.autotune import TuneProfile

    profile = TuneProfile(cpu_score=42)
    apply = Mock(return_value={"applied": True})
    monkeypatch.setattr("pyaccelerate.autotune.auto_tune", lambda **_kwargs: profile)
    monkeypatch.setattr("pyaccelerate.autotune.apply_profile", apply)
    cli.main(["tune", "--json", "--apply"])
    captured = capsys.readouterr()
    assert json.loads(captured.out) == asdict(profile)
    apply.assert_called_once_with(profile)
    assert "Profile applied" in captured.err
