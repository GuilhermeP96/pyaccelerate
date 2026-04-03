"""
pyaccelerate.rate_limiter — Token-bucket rate limiter.

Provides thread-safe rate limiting for API calls, I/O operations, or any
callable that must respect a throughput ceiling.

Usage::

    from pyaccelerate.rate_limiter import RateLimiter

    limiter = RateLimiter(rate=10, burst=20)  # 10 ops/sec, burst of 20

    for item in items:
        limiter.acquire()        # blocks until a token is available
        process(item)

    # Or as a decorator
    @limiter.wrap
    def api_call(url):
        ...
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Callable, Optional, TypeVar

log = logging.getLogger("pyaccelerate.rate_limiter")

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


class RateLimiter:
    """Thread-safe token-bucket rate limiter.

    Parameters
    ----------
    rate
        Tokens per second (sustained throughput).
    burst
        Maximum burst size (token bucket capacity).
    """

    def __init__(self, rate: float, burst: int = 0):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.burst = max(burst, 1)
        self._tokens = float(self.burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._total_acquired = 0
        self._total_waited_s = 0.0

    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Block until *tokens* are available, then consume them.

        Parameters
        ----------
        tokens
            Number of tokens to consume.
        timeout
            Maximum seconds to wait. None = wait forever. 0 = non-blocking.

        Returns
        -------
        bool
            True if tokens were acquired, False on timeout.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        wait_start = time.monotonic()

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._total_acquired += tokens
                    self._total_waited_s += time.monotonic() - wait_start
                    return True

                # Calculate wait time
                deficit = tokens - self._tokens
                wait = deficit / self.rate

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)

            time.sleep(min(wait, 0.1))

    def try_acquire(self, tokens: int = 1) -> bool:
        """Non-blocking acquire. Returns True if tokens were available."""
        return self.acquire(tokens=tokens, timeout=0)

    def _refill(self) -> None:
        """Add tokens based on elapsed time (called under lock)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            new_tokens = elapsed * self.rate
            self._tokens = min(self._tokens + new_tokens, float(self.burst))
            self._last_refill = now

    def wrap(self, fn: F) -> F:
        """Decorator that rate-limits calls to *fn*.

        Usage::

            @limiter.wrap
            def api_call(url):
                ...
        """
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.acquire()
            return fn(*args, **kwargs)
        return wrapper  # type: ignore[return-value]

    @property
    def available_tokens(self) -> float:
        """Current number of available tokens."""
        with self._lock:
            self._refill()
            return self._tokens

    def stats(self) -> dict:
        """Return rate limiter statistics."""
        with self._lock:
            self._refill()
            return {
                "rate": self.rate,
                "burst": self.burst,
                "available_tokens": round(self._tokens, 2),
                "total_acquired": self._total_acquired,
                "total_waited_s": round(self._total_waited_s, 4),
            }
