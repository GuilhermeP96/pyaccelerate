from __future__ import annotations

import pytest

from pyaccelerate.spark import configure_spark_builder, configure_spark_session


@pytest.mark.spark
def test_local_spark_session_smoke():
    pyspark = pytest.importorskip("pyspark")
    builder = pyspark.sql.SparkSession.builder.master("local[1]").appName("pyaccelerate-compat")
    builder = configure_spark_builder(
        builder,
        spark_version=pyspark.__version__,
        default_parallelism=1,
    )
    spark = builder.getOrCreate()
    try:
        applied = configure_spark_session(spark, default_parallelism=1)
        assert spark.range(5).count() == 5
        assert applied["spark.sql.adaptive.enabled"] == "true"
        assert spark.conf.get("spark.sql.shuffle.partitions") == "2"
    finally:
        spark.stop()
