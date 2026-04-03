"""
pyaccelerate.retry — Configurable retry with exponential backoff & jitter.

Provides decorators and context managers for retrying flaky operations:

  - ``@retry()``          — decorator with configurable backoff
  - ``retry_call()``      — functional retry wrapper
  - ``RetryPolicy``       — reusable policy object

Supports:
  - Exponential backoff with configurable base and cap
  - Full jitter (AWS-style) to prevent thundering herd
  - Per-exception filtering (only retry specific exception types)
  - Hooks for logging / metrics on each retry

Usage::

    from pyaccelerate.retry import retry, RetryPolicy

    @retry(max_attempts=3, backoff_base=0.5, retryable=(IOError, TimeoutError))
    def flaky_download(url):
        ...

    # Or with a reusable policy
    policy = RetryPolicy(max_attempts=5, backoff_base=1.0, backoff_cap=30.0)
    result = policy.call(flaky_download, url)
"""

from __future__ import annotations

import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple, Type, TypeVar

log = logging.getLogger("pyaccelerate.retry")

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

# Default retryable exceptions
_DEFAULT_RETRYABLE: Tuple[Type[BaseException], ...] = (
    IOError,
    OSError,
    TimeoutError,
    ConnectionError,
)


@dataclass
class RetryStats:
    """Statistics from a retry execution."""
    attempts: int = 0
    total_backoff_s: float = 0.0
    last_exception: Optional[BaseException] = None
    succeeded: bool = False


@dataclass
class RetryPolicy:
    """Reusable retry configuration.

    Parameters
    ----------
    max_attempts
        Maximum number of tries (including the first).
    backoff_base
        Base delay in seconds for exponential backoff.
    backoff_cap
        Maximum delay in seconds.
    backoff_factor
        Multiplier for each successive retry.
    jitter
        If True, apply full jitter (random [0, backoff]).
    retryable
        Tuple of exception types to retry on.
    on_retry
        Optional callback ``(attempt, exception, delay)`` called before each retry.
    """
    max_attempts: int = 3
    backoff_base: float = 0.5
    backoff_cap: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retryable: Tuple[Type[BaseException], ...] = _DEFAULT_RETRYABLE
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None

    def delay_for(self, attempt: int) -> float:
        """Compute the delay in seconds for the given attempt number (1-based)."""
        raw = min(
            self.backoff_base * (self.backoff_factor ** (attempt - 1)),
            self.backoff_cap,
        )
        if self.jitter:
            return random.uniform(0, raw)
        return raw

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call *fn* with retry logic applied.

        Raises the last exception if all attempts fail.
        """
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except self.retryable as exc:
                last_exc = exc
                if attempt >= self.max_attempts:
                    break
                delay = self.delay_for(attempt)
                if self.on_retry is not None:
                    self.on_retry(attempt, exc, delay)
                else:
                    log.warning(
                        "Retry %d/%d after %.2fs: %s",
                        attempt, self.max_attempts, delay, exc,
                    )
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]


def retry(
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    backoff_cap: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable: Tuple[Type[BaseException], ...] = _DEFAULT_RETRYABLE,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Callable[[F], F]:
    """Decorator that retries a function with exponential backoff.

    Usage::

        @retry(max_attempts=3, backoff_base=1.0)
        def flaky_io():
            ...
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        backoff_base=backoff_base,
        backoff_cap=backoff_cap,
        backoff_factor=backoff_factor,
        jitter=jitter,
        retryable=retryable,
        on_retry=on_retry,
    )

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return policy.call(fn, *args, **kwargs)
        return wrapper  # type: ignore[return-value]

    return decorator


def retry_call(
    fn: Callable[..., T],
    args: Sequence[Any] = (),
    kwargs: Optional[dict] = None,
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    retryable: Tuple[Type[BaseException], ...] = _DEFAULT_RETRYABLE,
) -> T:
    """Functional retry wrapper (no decorator needed).

    Usage::

        result = retry_call(download_file, args=(url,), max_attempts=5)
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        backoff_base=backoff_base,
        retryable=retryable,
    )
    return policy.call(fn, *args, **(kwargs or {}))
