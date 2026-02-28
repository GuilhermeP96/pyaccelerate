"""Tests for pyaccelerate.threads module."""

import time
from concurrent.futures import Future

from pyaccelerate.threads import (
    get_pool,
    io_pool_size,
    submit,
    map_parallel,
    run_parallel,
    run_parallel_collect,
    batch_execute,
    shutdown_pools,
)


class TestPool:
    def test_pool_creation(self):
        pool = get_pool()
        assert pool is not None

    def test_pool_is_singleton(self):
        a = get_pool()
        b = get_pool()
        assert a is b

    def test_io_pool_size_positive(self):
        size = io_pool_size()
        assert 4 <= size <= 32


class TestSubmit:
    def test_submit_returns_future(self):
        fut = submit(lambda x: x * 2, 21)
        assert isinstance(fut, Future)
        assert fut.result(timeout=5) == 42

    def test_submit_exception(self):
        def fail():
            raise ValueError("boom")

        fut = submit(fail)
        try:
            fut.result(timeout=5)
            assert False, "Expected ValueError"
        except ValueError:
            pass


class TestMapParallel:
    def test_basic(self):
        results = map_parallel(lambda x: x ** 2, [(i,) for i in range(10)])
        assert results == [i ** 2 for i in range(10)]

    def test_empty(self):
        results = map_parallel(lambda x: x, [])
        assert results == []


class TestRunParallel:
    def test_completes_all(self):
        results = []

        def collect(x):
            results.append(x)

        count = run_parallel(
            collect,
            [(i,) for i in range(20)],
            max_concurrent=4,
        )
        assert count == 20

    def test_with_error_handler(self):
        errors = []

        def fail_on_5(x):
            if x == 5:
                raise RuntimeError("five!")
            return x

        count = run_parallel(
            fail_on_5,
            [(i,) for i in range(10)],
            max_concurrent=3,
            on_error=lambda exc, args: errors.append(args),
        )
        assert count == 10
        assert len(errors) == 1


class TestBatchExecute:
    def test_basic(self):
        results = batch_execute(
            lambda x: x * 2,
            [(i,) for i in range(5)],
            max_concurrent=3,
            show_progress=False,
        )
        assert sorted(results) == [0, 2, 4, 6, 8]
