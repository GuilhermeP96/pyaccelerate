"""
pyaccelerate.lockfree_queue — High-performance lock-free task queues.

Provides two queue implementations optimized for work-stealing schedulers:

- **WorkDeque**: Per-worker Chase-Lev double-ended queue. The owner thread
  pushes / pops from the bottom (LIFO — cache-friendly, no lock). Stealers
  take from the top (FIFO) through a lightweight spinlock.
- **MPMCQueue**: Multi-Producer Multi-Consumer global injection queue backed
  by ``collections.deque`` (C-implemented, GIL-protected for append/popleft).

Design rationale
----------------
Under CPython's GIL, ``collections.deque.append`` / ``.popleft`` are atomic
at the bytecode level, giving us *effectively* lock-free behaviour for the
common case.  The only explicit lock is the steal-side spinlock on the
per-worker deque — and even that is only contested when a worker's own queue
is empty (rare under healthy load).

Inspired by:
- Chase-Lev deque (Tokio, Java ForkJoinPool)
- Go runtime's per-P run queues
"""

from __future__ import annotations

import collections
import threading
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════
#  Per-worker Chase-Lev style deque
# ═══════════════════════════════════════════════════════════════════════════

class WorkDeque(Generic[T]):
    """Per-worker double-ended queue with lock-free owner operations.

    * **Owner** thread: ``push()`` / ``pop()`` from the *bottom* (LIFO).
      Under CPython the GIL guarantees atomicity of ``deque.append`` and
      ``deque.pop``, so no lock is needed on the owner side.
    * **Stealers**: ``steal()`` / ``steal_batch()`` from the *top* (FIFO).
      A lightweight lock protects only the steal path; contention is rare
      because steals only happen when the stealer's own queue is empty.

    This mirrors the Chase-Lev deque used by Tokio and Java ForkJoinPool.
    """

    __slots__ = ("_deque", "_steal_lock", "steal_count", "push_count")

    def __init__(self) -> None:
        self._deque: collections.deque[T] = collections.deque()
        self._steal_lock = threading.Lock()
        self.steal_count: int = 0
        self.push_count: int = 0

    # ── Owner side (lock-free under GIL) ─────────────────────────────

    def push(self, item: T) -> None:
        """Push to bottom (owner thread — no lock)."""
        self._deque.append(item)
        self.push_count += 1

    def pop(self) -> Optional[T]:
        """Pop from bottom (owner thread — LIFO, cache-friendly)."""
        try:
            return self._deque.pop()
        except IndexError:
            return None

    # ── Steal side ───────────────────────────────────────────────────

    def steal(self) -> Optional[T]:
        """Steal one item from the top (other workers — FIFO)."""
        if not self._deque:  # fast non-locking check
            return None
        with self._steal_lock:
            try:
                item = self._deque.popleft()
                self.steal_count += 1
                return item
            except IndexError:
                return None

    def steal_batch(self, max_items: int) -> List[T]:
        """Steal up to *max_items* from the top in one lock acquisition.

        Takes at most half the deque to keep the owner fed.
        """
        if not self._deque:
            return []
        with self._steal_lock:
            n = min(max_items, len(self._deque) // 2 or 1)
            batch: List[T] = []
            for _ in range(n):
                try:
                    batch.append(self._deque.popleft())
                except IndexError:
                    break
            self.steal_count += len(batch)
            return batch

    # ── Introspection ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._deque)

    @property
    def is_empty(self) -> bool:
        return len(self._deque) == 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "length": len(self._deque),
            "push_count": self.push_count,
            "steal_count": self.steal_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Global injection MPMC queue
# ═══════════════════════════════════════════════════════════════════════════

class MPMCQueue(Generic[T]):
    """Multi-Producer Multi-Consumer queue for global task injection.

    Backed by ``collections.deque`` (C-implemented).  A ``threading.Event``
    allows consumer threads to sleep efficiently when the queue is empty,
    rather than busy-spinning.
    """

    __slots__ = ("_deque", "_not_empty", "_maxsize")

    def __init__(self, maxsize: int = 0) -> None:
        self._deque: collections.deque[T] = collections.deque(
            maxlen=maxsize or None,
        )
        self._not_empty = threading.Event()
        self._maxsize = maxsize

    # ── Put ───────────────────────────────────────────────────────────

    def put(self, item: T) -> None:
        """Enqueue a single item."""
        self._deque.append(item)
        self._not_empty.set()

    def put_batch(self, items: List[T]) -> None:
        """Enqueue a batch of items."""
        self._deque.extend(items)
        if items:
            self._not_empty.set()

    # ── Get ───────────────────────────────────────────────────────────

    def get(self) -> Optional[T]:
        """Dequeue a single item (non-blocking). Returns ``None`` if empty."""
        try:
            item = self._deque.popleft()
            if not self._deque:
                self._not_empty.clear()
            return item
        except IndexError:
            return None

    def get_batch(self, max_items: int) -> List[T]:
        """Dequeue up to *max_items* in one call."""
        batch: List[T] = []
        for _ in range(max_items):
            try:
                batch.append(self._deque.popleft())
            except IndexError:
                break
        if not self._deque:
            self._not_empty.clear()
        return batch

    # ── Wait ──────────────────────────────────────────────────────────

    def wait(self, timeout: float = 1.0) -> bool:
        """Block until items are available or *timeout* expires."""
        return self._not_empty.wait(timeout)

    def signal(self) -> None:
        """Wake up any waiting consumers."""
        self._not_empty.set()

    # ── Introspection ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._deque)

    @property
    def is_empty(self) -> bool:
        return len(self._deque) == 0
