"""Apache Spark integration helpers.

The integration is intentionally dependency-optional: importing
``pyaccelerate.spark`` never imports PySpark.  Applications may therefore use
the core package on drivers and executors where PySpark is supplied by the
cluster rather than installed from PyPI.

Spark configuration is split between builder-time and runtime-safe settings.
``spark.python.worker.reuse`` must be configured before the Spark context is
created, while the SQL settings can also be applied to an existing session.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, Mapping

from packaging.version import InvalidVersion, Version

from pyaccelerate.cpu import recommend_workers

MIN_SPARK_VERSION = Version("3.2")


class SparkCompatibilityError(RuntimeError):
    """Raised when PySpark is unavailable or older than the supported floor."""


def _raw_spark_version(spark: Any | None = None) -> str:
    if spark is not None:
        raw = getattr(spark, "version", None)
        if raw:
            return str(raw)

    try:
        return package_version("pyspark")
    except PackageNotFoundError:
        # Some managed clusters add PySpark to PYTHONPATH without installing
        # distribution metadata.  Keep that deployment mode supported.
        try:
            pyspark = import_module("pyspark")
        except ImportError as exc:
            raise SparkCompatibilityError(
                "PySpark is not available; install pyaccelerate[spark] or pass a Spark session"
            ) from exc
        raw = getattr(pyspark, "__version__", None)
        if not raw:
            raise SparkCompatibilityError(
                "Unable to determine the installed Spark version"
            ) from None
        return str(raw)


def ensure_spark_compatible(spark: Any | None = None) -> Version:
    """Return the detected Spark version after enforcing the 3.2+ contract.

    ``spark`` is duck-typed so the helper works with classic ``SparkSession``
    instances and managed-cluster proxies without importing PySpark eagerly.
    """

    raw = _raw_spark_version(spark)
    try:
        detected = Version(raw)
    except InvalidVersion as exc:
        raise SparkCompatibilityError(f"Invalid Spark version: {raw!r}") from exc

    if detected < MIN_SPARK_VERSION:
        raise SparkCompatibilityError(
            f"Spark {detected} is unsupported; pyaccelerate requires Spark 3.2+"
        )
    return detected


def _default_parallelism(spark: Any | None) -> int:
    if spark is not None:
        context = getattr(spark, "sparkContext", None)
        cluster_default = getattr(context, "defaultParallelism", None)
        if isinstance(cluster_default, int) and cluster_default > 0:
            return cluster_default

    # Without a live context this is only a local fallback.  The result remains
    # subordinate to Spark's own cluster-derived default when a session exists.
    return max(1, recommend_workers(io_bound=False))


def recommend_spark_config(
    spark: Any | None = None,
    *,
    default_parallelism: int | None = None,
    partition_multiplier: int = 2,
) -> dict[str, str]:
    """Return conservative Spark 3.2+ settings for Python workloads.

    The recommendation enables Python worker reuse and adaptive query
    execution.  Shuffle partitions are derived from the live cluster's
    ``defaultParallelism`` when available, never from driver hardware alone.
    Explicit values make the function deterministic for deployment tooling.
    """

    ensure_spark_compatible(spark)
    if partition_multiplier < 1:
        raise ValueError("partition_multiplier must be at least 1")
    if default_parallelism is not None and default_parallelism < 1:
        raise ValueError("default_parallelism must be at least 1")

    parallelism = default_parallelism or _default_parallelism(spark)
    return {
        "spark.python.worker.reuse": "true",
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.shuffle.partitions": str(parallelism * partition_multiplier),
    }


def configure_spark_builder(
    builder: Any,
    *,
    spark_version: str,
    config: Mapping[str, Any] | None = None,
    default_parallelism: int | None = None,
    partition_multiplier: int = 2,
) -> Any:
    """Apply pyaccelerate recommendations to a ``SparkSession.Builder``.

    ``spark_version`` is explicit because a builder has no active session from
    which to read it.  The same builder (or the replacement returned by a
    compatible proxy) is returned for fluent use.
    """

    class _VersionProbe:
        version = spark_version

    options = recommend_spark_config(
        _VersionProbe(),
        default_parallelism=default_parallelism,
        partition_multiplier=partition_multiplier,
    )
    if config:
        options.update({str(key): str(value) for key, value in config.items()})

    configured = builder
    for key, value in options.items():
        replacement = configured.config(key, value)
        if replacement is not None:
            configured = replacement
    return configured


def configure_spark_session(
    spark: Any,
    *,
    config: Mapping[str, Any] | None = None,
    default_parallelism: int | None = None,
    partition_multiplier: int = 2,
) -> dict[str, str]:
    """Apply runtime-safe recommendations to an existing Spark session.

    Builder-only settings such as Python worker reuse are deliberately omitted
    because Spark cannot reliably change them after the context has started.
    The returned mapping contains exactly the settings applied.
    """

    options = recommend_spark_config(
        spark,
        default_parallelism=default_parallelism,
        partition_multiplier=partition_multiplier,
    )
    options.pop("spark.python.worker.reuse", None)
    if config:
        options.update({str(key): str(value) for key, value in config.items()})

    for key, value in options.items():
        spark.conf.set(key, value)
    return options


__all__ = [
    "MIN_SPARK_VERSION",
    "SparkCompatibilityError",
    "configure_spark_builder",
    "configure_spark_session",
    "ensure_spark_compatible",
    "recommend_spark_config",
]
