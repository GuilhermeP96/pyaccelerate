from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest
from packaging.version import Version

from pyaccelerate.spark import (
    SparkCompatibilityError,
    configure_spark_builder,
    configure_spark_session,
    ensure_spark_compatible,
    recommend_spark_config,
)


class _Context:
    defaultParallelism = 6  # noqa: N815 - mirrors PySpark's public API


class _Conf:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value


class _Session:
    version = "3.2.0"
    sparkContext = _Context()  # noqa: N815 - mirrors PySpark's public API

    def __init__(self):
        self.conf = _Conf()


class _Builder:
    def __init__(self):
        self.values = {}

    def config(self, key, value):
        self.values[key] = value
        return self


@pytest.mark.parametrize("version", ["3.2.0", "3.5.8", "4.3.0"])
def test_supported_spark_versions(version):
    session = _Session()
    session.version = version
    assert ensure_spark_compatible(session) == Version(version)


def test_spark_older_than_32_is_rejected():
    session = _Session()
    session.version = "3.1.3"
    with pytest.raises(SparkCompatibilityError, match=r"requires Spark 3\.2"):
        ensure_spark_compatible(session)


def test_missing_pyspark_has_actionable_error():
    with patch(  # noqa: SIM117 - nested form remains valid on Python 3.8
        "pyaccelerate.spark.package_version", side_effect=PackageNotFoundError
    ), patch("pyaccelerate.spark.import_module", side_effect=ImportError):
        with pytest.raises(SparkCompatibilityError, match=r"pyaccelerate\[spark\]"):
            ensure_spark_compatible()


def test_recommendation_uses_cluster_parallelism():
    config = recommend_spark_config(_Session(), partition_multiplier=3)
    assert config == {
        "spark.python.worker.reuse": "true",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.shuffle.partitions": "18",
    }


def test_builder_receives_static_and_custom_config():
    builder = _Builder()
    returned = configure_spark_builder(
        builder,
        spark_version="3.2.0",
        default_parallelism=4,
        config={"spark.app.name": "accelerated"},
    )
    assert returned is builder
    assert builder.values["spark.python.worker.reuse"] == "true"
    assert builder.values["spark.sql.shuffle.partitions"] == "8"
    assert builder.values["spark.app.name"] == "accelerated"


def test_existing_session_only_receives_runtime_safe_config():
    session = _Session()
    applied = configure_spark_session(session)
    assert "spark.python.worker.reuse" not in applied
    assert session.conf.values == applied
    assert applied["spark.sql.shuffle.partitions"] == "12"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"default_parallelism": 0},
        {"partition_multiplier": 0},
    ],
)
def test_invalid_partition_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        recommend_spark_config(_Session(), **kwargs)
