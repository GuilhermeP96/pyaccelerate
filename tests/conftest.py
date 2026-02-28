"""Shared fixtures for pyaccelerate tests."""

import pytest


@pytest.fixture(autouse=True)
def _reset_caches():
    """Reset all singleton caches between tests."""
    yield
    # GPU
    try:
        from pyaccelerate.gpu.detector import reset_cache as gpu_reset
        gpu_reset()
    except Exception:
        pass
    # Virt
    try:
        from pyaccelerate.virt import reset_cache as virt_reset
        virt_reset()
    except Exception:
        pass
    # CPU
    try:
        import pyaccelerate.cpu as cpu_mod
        cpu_mod._cached_info = None
    except Exception:
        pass
