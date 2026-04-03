"""
pyaccelerate.circuit_breaker — Circuit breaker pattern for flaky operations.

Prevents cascading failures by tripping a circuit after a threshold of
consecutive errors, then periodically allowing a probe to check recovery.

States:
  - **CLOSED**: Normal operation — all calls pass through.
  - **OPEN**: Tripped — calls are immediately rejected (``CircuitOpenError``).
  - **HALF_OPEN**: Probing — one call is allowed to test recovery.

Usage::

    from pyaccelerate.circuit_breaker import CircuitBreaker, CircuitOpenError

    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

    try:
        result = breaker.call(flaky_api_call, param1, param2)
    except CircuitOpenError:
        result = cached_fallback()
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

log = logging.getLogger("pyaccelerate.circuit_breaker")

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker state machine."""
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open (rejecting calls)."""

    def __init__(self, breaker_name: str, retry_after_s: float):
        self.breaker_name = breaker_name
        self.retry_after_s = retry_after_s
        super().__init__(
            f"Circuit '{breaker_name}' is OPEN — retry after {retry_after_s:.1f}s"
        )


class CircuitBreaker:
    """Thread-safe circuit breaker.

    Parameters
    ----------
    failure_threshold
        Number of consecutive failures before the circuit opens.
    recovery_timeout
        Seconds to wait before allowing a probe (half-open).
    monitored_exceptions
        Tuple of exception types that count as failures.
    name
        Human-readable name for logging/metrics.
    on_state_change
        Optional callback ``(old_state, new_state)`` on transitions.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        monitored_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
        name: str = "default",
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.monitored_exceptions = monitored_exceptions
        self.name = name
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._transition(CircuitState.HALF_OPEN)
            return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute *fn* through the circuit breaker.

        Raises ``CircuitOpenError`` if the circuit is open.
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._transition(CircuitState.HALF_OPEN)
                else:
                    raise CircuitOpenError(self.name, self.recovery_timeout - elapsed)

        try:
            result = fn(*args, **kwargs)
        except self.monitored_exceptions as exc:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def reset(self) -> None:
        """Force-reset the circuit to CLOSED."""
        with self._lock:
            old = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            if old != CircuitState.CLOSED:
                log.info("Circuit '%s' force-reset to CLOSED", self.name)

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._transition(CircuitState.OPEN)
            elif (
                self._state == CircuitState.CLOSED
                and self._failure_count >= self.failure_threshold
            ):
                self._transition(CircuitState.OPEN)

    def _on_success(self) -> None:
        with self._lock:
            self._success_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._transition(CircuitState.CLOSED)
                self._failure_count = 0

    def _transition(self, new_state: CircuitState) -> None:
        """Transition to a new state (must hold _lock)."""
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        log.info("Circuit '%s': %s → %s", self.name, old.name, new_state.name)
        if self.on_state_change:
            try:
                self.on_state_change(old, new_state)
            except Exception:
                pass

    def as_dict(self) -> dict:
        """Return the breaker state as a dict."""
        return {
            "name": self.name,
            "state": self.state.name,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_s": self.recovery_timeout,
        }
