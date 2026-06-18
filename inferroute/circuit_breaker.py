"""
Circuit Breaker implementation for InferRoute.

Implements a per-backend state machine: CLOSED → OPEN → HALF_OPEN → CLOSED.
State is persisted in Redis so it is shared across multiple gateway replicas.

States:
    CLOSED    — normal operation; failures are counted.
    OPEN      — backend is bypassed for `recovery_timeout` seconds.
    HALF_OPEN — one probe request is allowed through. Success → CLOSED;
                failure → OPEN (resets timer).

Usage:
    cb = CircuitBreaker("openai", redis_client)
    if not await cb.allow_request():
        raise HTTPException(503, "Circuit open — backend unavailable")
    try:
        result = await adapter.generate(req)
        await cb.record_success()
    except Exception as e:
        await cb.record_failure()
        raise
"""
import time
import logging
from enum import Enum
from typing import Optional

from inferroute.config import settings
from inferroute.observability import (
    CIRCUIT_BREAKER_STATE,
    CIRCUIT_BREAKER_TRIP_TOTAL,
)

logger = logging.getLogger("inferroute.circuit_breaker")


class CBState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    Redis-backed per-backend circuit breaker.
    All state is stored in Redis keys so multiple replicas share the same view.
    """

    def __init__(self, backend: str, redis_client=None):
        self.backend = backend
        self._redis = redis_client  # injected after startup

        # Config from settings
        self.failure_threshold: int = settings.CB_FAILURE_THRESHOLD
        self.recovery_timeout: int = settings.CB_RECOVERY_TIMEOUT_S
        self.success_threshold: int = settings.CB_SUCCESS_THRESHOLD

        # Redis key names
        self._k_state = f"inferroute:cb:{backend}:state"
        self._k_fail_count = f"inferroute:cb:{backend}:fail_count"
        self._k_success_count = f"inferroute:cb:{backend}:success_count"
        self._k_opened_at = f"inferroute:cb:{backend}:opened_at"

    def set_redis(self, redis_client) -> None:
        """Inject the Redis client after lifespan startup."""
        self._redis = redis_client

    async def _get_state(self) -> CBState:
        """Read current state from Redis (defaults to CLOSED if absent)."""
        if self._redis is None:
            return CBState.CLOSED
        raw = await self._redis.get(self._k_state)
        if raw is None:
            return CBState.CLOSED
        return CBState(raw)

    async def _set_state(self, state: CBState) -> None:
        if self._redis is None:
            return
        await self._redis.set(self._k_state, state.value)
        # Update Prometheus gauge: 0=CLOSED, 1=OPEN, 2=HALF_OPEN
        gauge_val = {CBState.CLOSED: 0, CBState.OPEN: 1, CBState.HALF_OPEN: 2}[state]
        CIRCUIT_BREAKER_STATE.labels(backend=self.backend).set(gauge_val)

    async def allow_request(self) -> bool:
        """
        Returns True if the request should be allowed through, False to reject.

        CLOSED   → always allow
        OPEN     → allow only if recovery_timeout has elapsed (→ HALF_OPEN)
        HALF_OPEN → allow (the single probe request)
        """
        if self._redis is None:
            return True  # fail-open when Redis is unavailable

        try:
            state = await self._get_state()

            if state == CBState.CLOSED:
                return True

            if state == CBState.OPEN:
                opened_at_raw = await self._redis.get(self._k_opened_at)
                if opened_at_raw is None:
                    return True  # stale state; assume closed
                elapsed = time.time() - float(opened_at_raw)
                if elapsed >= self.recovery_timeout:
                    logger.info(f"[CB:{self.backend}] Recovery timeout elapsed — entering HALF_OPEN")
                    await self._set_state(CBState.HALF_OPEN)
                    await self._redis.set(self._k_success_count, "0")
                    return True
                return False  # still open

            if state == CBState.HALF_OPEN:
                return True  # allow the probe

        except Exception as e:
            logger.warning(f"[CB:{self.backend}] Redis error in allow_request: {e} — failing open")
            return True

        return True

    async def record_success(self) -> None:
        """Record a successful request. In HALF_OPEN, accumulate successes toward recovery."""
        if self._redis is None:
            return

        try:
            state = await self._get_state()

            if state == CBState.HALF_OPEN:
                count = await self._redis.incr(self._k_success_count)
                if int(count) >= self.success_threshold:
                    logger.info(f"[CB:{self.backend}] Recovered — returning to CLOSED")
                    await self._close()
            elif state == CBState.CLOSED:
                # Reset failure counter on success
                await self._redis.set(self._k_fail_count, "0")

        except Exception as e:
            logger.warning(f"[CB:{self.backend}] Redis error in record_success: {e}")

    async def record_failure(self) -> None:
        """Record a failure. Transition CLOSED→OPEN when threshold is reached."""
        if self._redis is None:
            return

        try:
            state = await self._get_state()

            if state in (CBState.CLOSED, CBState.HALF_OPEN):
                if state == CBState.HALF_OPEN:
                    logger.warning(f"[CB:{self.backend}] Probe failed — returning to OPEN")
                    await self._open()
                    return

                count = await self._redis.incr(self._k_fail_count)
                logger.debug(f"[CB:{self.backend}] Failure #{count}/{self.failure_threshold}")
                if int(count) >= self.failure_threshold:
                    logger.warning(f"[CB:{self.backend}] Failure threshold reached — OPENING circuit")
                    await self._open()

        except Exception as e:
            logger.warning(f"[CB:{self.backend}] Redis error in record_failure: {e}")

    async def _open(self) -> None:
        """Transition to OPEN state."""
        await self._set_state(CBState.OPEN)
        await self._redis.set(self._k_opened_at, str(time.time()))
        await self._redis.set(self._k_fail_count, "0")
        await self._redis.set(self._k_success_count, "0")
        CIRCUIT_BREAKER_TRIP_TOTAL.labels(backend=self.backend).inc()
        logger.warning(f"[CB:{self.backend}] Circuit OPENED")

    async def _close(self) -> None:
        """Transition to CLOSED state."""
        await self._set_state(CBState.CLOSED)
        await self._redis.set(self._k_fail_count, "0")
        await self._redis.set(self._k_success_count, "0")
        logger.info(f"[CB:{self.backend}] Circuit CLOSED (healthy)")

    async def get_status(self) -> dict:
        """Return a dict summary for the /v1/routing/status endpoint."""
        if self._redis is None:
            return {"backend": self.backend, "state": "CLOSED", "fail_count": 0, "redis_available": False}

        try:
            state = await self._get_state()
            fail_count = int(await self._redis.get(self._k_fail_count) or 0)
            opened_at_raw = await self._redis.get(self._k_opened_at)
            opened_at = float(opened_at_raw) if opened_at_raw else None
            time_until_probe = None
            if state == CBState.OPEN and opened_at:
                remaining = self.recovery_timeout - (time.time() - opened_at)
                time_until_probe = max(0.0, remaining)

            return {
                "backend": self.backend,
                "state": state.value,
                "fail_count": fail_count,
                "failure_threshold": self.failure_threshold,
                "time_until_probe_s": time_until_probe,
                "redis_available": True,
            }
        except Exception as e:
            return {"backend": self.backend, "state": "UNKNOWN", "error": str(e)}


# ── Module-level registry ────────────────────────────────────────────────────
# Initialized with None redis; redis is injected at startup in main.py

_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(backend: str) -> CircuitBreaker:
    """Return (or lazily create) the circuit breaker for a backend."""
    if backend not in _breakers:
        _breakers[backend] = CircuitBreaker(backend)
    return _breakers[backend]


def initialize_circuit_breakers(redis_client, backends: list[str]) -> None:
    """Called at startup to inject Redis into all circuit breakers."""
    for backend in backends:
        cb = get_circuit_breaker(backend)
        cb.set_redis(redis_client)
    logger.info(f"Circuit breakers initialized for backends: {backends}")
