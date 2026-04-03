"""
pyaccelerate.plugins — Lightweight plugin discovery via entry points.

Register plugins in ``pyproject.toml``::

    [project.entry-points."pyaccelerate.plugins"]
    my_plugin = "my_package:setup"

Discover and load them::

    from pyaccelerate.plugins import discover, register, get_all

    discover()                         # auto-load entry-point plugins
    register("manual", my_hook_fn)     # manual registration
    for name, fn in get_all():
        fn(engine)
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("pyaccelerate.plugins")

ENTRY_POINT_GROUP = "pyaccelerate.plugins"

PluginFn = Callable[..., Any]

_registry: Dict[str, PluginFn] = {}


def register(name: str, fn: PluginFn) -> None:
    """Register a plugin by name."""
    if name in _registry:
        log.warning("Plugin %r already registered — overwriting", name)
    _registry[name] = fn
    log.debug("Registered plugin %r", name)


def unregister(name: str) -> Optional[PluginFn]:
    """Remove a plugin. Returns the removed callable or None."""
    return _registry.pop(name, None)


def get(name: str) -> Optional[PluginFn]:
    """Lookup a single plugin by name."""
    return _registry.get(name)


def get_all() -> List[Tuple[str, PluginFn]]:
    """Return all registered plugins as (name, fn) pairs."""
    return list(_registry.items())


def discover() -> int:
    """Auto-discover plugins registered via entry points.

    Returns the number of plugins successfully loaded.
    """
    loaded = 0
    try:
        eps = importlib.metadata.entry_points()
        group = eps.get(ENTRY_POINT_GROUP, []) if isinstance(eps, dict) else [
            ep for ep in eps if ep.group == ENTRY_POINT_GROUP
        ]
    except Exception:
        log.debug("entry_points() lookup failed", exc_info=True)
        return 0

    for ep in group:
        try:
            fn = ep.load()
            register(ep.name, fn)
            loaded += 1
            log.info("Loaded plugin %r from %s", ep.name, ep.value)
        except Exception:
            log.warning("Failed to load plugin %r", ep.name, exc_info=True)

    return loaded


def clear() -> None:
    """Remove all registered plugins."""
    _registry.clear()
