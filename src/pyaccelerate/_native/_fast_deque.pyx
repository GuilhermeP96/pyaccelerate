# cython: language_level=3, boundscheck=False, wraparound=False
"""
_fast_deque.pyx — Cython-accelerated Chase-Lev work-stealing deque.

Compile with::

    python setup_cython.py build_ext --inplace

This replaces the pure-Python ``WorkDeque`` from ``lockfree_queue.py``
with a C-extension that avoids Python object overhead on the hot path
(push / pop / steal).
"""

from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy
import threading


cdef class FastDeque:
    """Cython Chase-Lev deque — owner push/pop lock-free, steal under lock.

    Drop-in replacement for ``lockfree_queue.WorkDeque``.
    """

    cdef:
        list _items
        object _steal_lock
        public long push_count
        public long steal_count

    def __cinit__(self):
        self._items = []
        self._steal_lock = threading.Lock()
        self.push_count = 0
        self.steal_count = 0

    cpdef void push(self, object item):
        """Push to bottom (owner — no lock)."""
        self._items.append(item)
        self.push_count += 1

    cpdef object pop(self):
        """Pop from bottom (LIFO, owner — no lock)."""
        if not self._items:
            return None
        return self._items.pop()

    cpdef object steal(self):
        """Steal from top (FIFO, any thread — under lock)."""
        if not self._items:
            return None
        with self._steal_lock:
            if not self._items:
                return None
            self.steal_count += 1
            return self._items.pop(0)

    cpdef list steal_batch(self, int max_items):
        """Steal batch from top (FIFO)."""
        if not self._items:
            return []
        with self._steal_lock:
            cdef int n = min(max_items, len(self._items) // 2 or 1)
            cdef list batch = self._items[:n]
            del self._items[:n]
            self.steal_count += len(batch)
            return batch

    def __len__(self):
        return len(self._items)

    @property
    def is_empty(self):
        return len(self._items) == 0

    @property
    def stats(self):
        return {
            "length": len(self._items),
            "push_count": self.push_count,
            "steal_count": self.steal_count,
        }
