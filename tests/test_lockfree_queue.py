"""Tests for pyaccelerate.lockfree_queue module."""

from pyaccelerate.lockfree_queue import WorkDeque, MPMCQueue


class TestWorkDeque:
    def test_push_pop_lifo(self):
        d: WorkDeque[int] = WorkDeque()
        d.push(1)
        d.push(2)
        d.push(3)
        assert d.pop() == 3
        assert d.pop() == 2
        assert d.pop() == 1
        assert d.pop() is None

    def test_steal_fifo(self):
        d: WorkDeque[int] = WorkDeque()
        d.push(10)
        d.push(20)
        d.push(30)
        assert d.steal() == 10
        assert d.steal() == 20
        assert d.steal() == 30
        assert d.steal() is None

    def test_steal_batch(self):
        d: WorkDeque[int] = WorkDeque()
        for i in range(10):
            d.push(i)
        batch = d.steal_batch(3)
        assert len(batch) <= 3
        assert all(isinstance(x, int) for x in batch)

    def test_empty(self):
        d: WorkDeque[str] = WorkDeque()
        assert d.is_empty
        assert len(d) == 0
        assert d.pop() is None
        assert d.steal() is None
        assert d.steal_batch(5) == []

    def test_stats(self):
        d: WorkDeque[int] = WorkDeque()
        d.push(1)
        d.push(2)
        d.steal()
        s = d.stats
        assert s["push_count"] == 2
        assert s["steal_count"] == 1
        assert s["length"] == 1

    def test_concurrent_push_steal(self):
        import threading

        d: WorkDeque[int] = WorkDeque()
        results = []
        errors = []

        def producer():
            for i in range(100):
                d.push(i)

        def stealer():
            stolen = 0
            for _ in range(200):
                item = d.steal()
                if item is not None:
                    stolen += 1
            results.append(stolen)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=stealer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Stealer got some items; remaining are still in deque
        total = results[0] + len(d)
        assert total <= 100


class TestMPMCQueue:
    def test_put_get(self):
        q: MPMCQueue[int] = MPMCQueue()
        q.put(1)
        q.put(2)
        assert q.get() == 1
        assert q.get() == 2
        assert q.get() is None

    def test_put_batch(self):
        q: MPMCQueue[int] = MPMCQueue()
        q.put_batch([10, 20, 30])
        assert len(q) == 3
        assert q.get() == 10

    def test_get_batch(self):
        q: MPMCQueue[int] = MPMCQueue()
        q.put_batch([1, 2, 3, 4, 5])
        batch = q.get_batch(3)
        assert batch == [1, 2, 3]
        assert len(q) == 2

    def test_empty(self):
        q: MPMCQueue[str] = MPMCQueue()
        assert q.is_empty
        assert len(q) == 0
        assert q.get() is None
        assert q.get_batch(5) == []

    def test_wait_returns_true_when_items(self):
        import threading

        q: MPMCQueue[int] = MPMCQueue()

        def delayed_put():
            import time
            time.sleep(0.05)
            q.put(42)

        t = threading.Thread(target=delayed_put)
        t.start()
        result = q.wait(timeout=2.0)
        assert result is True
        t.join()

    def test_wait_returns_false_on_timeout(self):
        q: MPMCQueue[int] = MPMCQueue()
        result = q.wait(timeout=0.05)
        assert result is False

    def test_maxsize(self):
        q: MPMCQueue[int] = MPMCQueue(maxsize=3)
        q.put(1)
        q.put(2)
        q.put(3)
        q.put(4)  # oldest drops
        assert len(q) == 3
        assert q.get() == 2  # 1 was dropped

    def test_concurrent_producers_consumers(self):
        import threading

        q: MPMCQueue[int] = MPMCQueue()
        produced = []
        consumed = []

        def producer(start: int):
            for i in range(50):
                q.put(start + i)
                produced.append(start + i)

        def consumer():
            for _ in range(200):
                item = q.get()
                if item is not None:
                    consumed.append(item)

        threads = [
            threading.Thread(target=producer, args=(0,)),
            threading.Thread(target=producer, args=(1000,)),
            threading.Thread(target=consumer),
            threading.Thread(target=consumer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All produced items should be consumed or still in queue
        total = len(consumed) + len(q)
        assert total == 100
