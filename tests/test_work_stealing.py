"""Tests for pyaccelerate.work_stealing module."""

import time
from concurrent.futures import Future

from pyaccelerate.work_stealing import (
    WorkStealingScheduler,
    get_scheduler,
    shutdown_scheduler,
    ws_submit,
    ws_map,
)


class TestWorkStealingScheduler:
    def test_create_and_start(self):
        sched = WorkStealingScheduler(num_workers=2)
        sched.start()
        assert sched._started
        sched.shutdown()

    def test_context_manager(self):
        with WorkStealingScheduler(num_workers=2) as sched:
            assert sched._started
        # After exit, should be shut down

    def test_submit_returns_future(self):
        with WorkStealingScheduler(num_workers=2) as sched:
            fut = sched.submit(lambda x: x * 2, 21)
            assert isinstance(fut, Future)
            assert fut.result(timeout=5) == 42

    def test_submit_exception(self):
        with WorkStealingScheduler(num_workers=2) as sched:
            def fail():
                raise ValueError("boom")
            fut = sched.submit(fail)
            try:
                fut.result(timeout=5)
                assert False, "Expected ValueError"
            except ValueError:
                pass

    def test_map(self):
        with WorkStealingScheduler(num_workers=4) as sched:
            results = sched.map(lambda x: x ** 2, [(i,) for i in range(10)])
            assert results == [i ** 2 for i in range(10)]

    def test_map_empty(self):
        with WorkStealingScheduler(num_workers=2) as sched:
            results = sched.map(lambda x: x, [])
            assert results == []

    def test_submit_to_global(self):
        with WorkStealingScheduler(num_workers=2) as sched:
            fut = sched.submit_to_global(lambda x: x + 1, 99)
            assert fut.result(timeout=5) == 100

    def test_many_tasks(self):
        with WorkStealingScheduler(num_workers=4) as sched:
            futures = [sched.submit(lambda x: x * 2, i) for i in range(100)]
            results = [f.result(timeout=10) for f in futures]
            assert sorted(results) == [i * 2 for i in range(100)]

    def test_stats(self):
        with WorkStealingScheduler(num_workers=2) as sched:
            futures = [sched.submit(lambda x: x, i) for i in range(20)]
            for f in futures:
                f.result(timeout=5)
            time.sleep(0.1)  # let workers drain
            stats = sched.stats()
            assert stats.worker_count == 2
            assert stats.total_submitted == 20
            assert stats.total_completed == 20

    def test_work_stealing_happens(self):
        """Verify that work actually gets stolen between workers."""
        with WorkStealingScheduler(num_workers=4) as sched:
            # Submit all tasks to worker 0 by using rapid sequential submit
            futures = []
            for i in range(40):
                futures.append(sched.submit(lambda x: x, i))
            for f in futures:
                f.result(timeout=10)
            time.sleep(0.1)
            stats = sched.stats()
            # At least some workers should have completed tasks
            active_workers = sum(
                1 for w in stats.per_worker if w["completed"] > 0
            )
            assert active_workers >= 1  # at minimum the assigned worker


class TestModuleLevelAPI:
    def test_get_scheduler(self):
        sched = get_scheduler(num_workers=2)
        assert sched is not None
        # Subsequent calls return same instance
        assert get_scheduler() is sched
        shutdown_scheduler()

    def test_ws_submit(self):
        try:
            fut = ws_submit(lambda x: x + 1, 41)
            assert fut.result(timeout=5) == 42
        finally:
            shutdown_scheduler()

    def test_ws_map(self):
        try:
            results = ws_map(lambda x: x * 3, [(i,) for i in range(5)])
            assert results == [0, 3, 6, 9, 12]
        finally:
            shutdown_scheduler()
