"""
pyaccelerate._native — Optional Cython / Rust accelerators.

This package provides optimised C-extension replacements for the hot-path
data structures in the scheduler.  When the native module is importable
the rest of *pyaccelerate* will use it automatically; otherwise the pure-
Python fallback is used transparently.

Build the Cython extension::

    pip install cython
    cd src/pyaccelerate/_native
    python setup_cython.py build_ext --inplace

Build the Rust extension (requires maturin + Rust)::

    cd bindings/rust/pyaccelerate_native
    maturin develop --release
"""

from __future__ import annotations

_NATIVE_AVAILABLE = False

try:
    from pyaccelerate._native._fast_deque import FastDeque as _FastDeque  # type: ignore[import-not-found]
    _NATIVE_AVAILABLE = True
except ImportError:
    _FastDeque = None

try:
    from pyaccelerate_native import FastDeque as _RustDeque  # type: ignore[import-not-found]
    _NATIVE_AVAILABLE = True
except ImportError:
    _RustDeque = None


def native_available() -> bool:
    """Return ``True`` if a compiled accelerator is importable."""
    return _NATIVE_AVAILABLE


def get_fast_deque_class() -> type | None:
    """Return the fastest available deque implementation, or ``None``."""
    if _RustDeque is not None:
        return _RustDeque
    if _FastDeque is not None:
        return _FastDeque
    return None
