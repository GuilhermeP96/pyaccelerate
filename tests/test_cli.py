"""Tests for machine-aware CLI dependency installation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from pyaccelerate import cli


def _engine(*, gpu: bool = False, usable_gpu: bool = False,
            npu: bool = False, usable_npu: bool = False) -> SimpleNamespace:
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
