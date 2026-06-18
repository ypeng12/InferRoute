"""
Unit tests for the Circuit Breaker state machine.

Tests:
  - CLOSED → OPEN on consecutive failures
  - OPEN rejects requests
  - OPEN → HALF_OPEN after recovery timeout
  - HALF_OPEN → CLOSED on successful probes
  - HALF_OPEN → OPEN on failed probe
  - Fail-open when Redis is unavailable
  - CB status dict
"""
import asyncio
import time
import pytest
from unittest.mock import AsyncMock, patch

from inferroute.circuit_breaker import CircuitBreaker, CBState


@pytest.fixture
def mock_redis():
    """In-memory dict-backed mock Redis for circuit breaker tests."""
    store: dict = {}
    r = AsyncMock()

    async def get(key):
        return store.get(key)

    async def set(key, value, *args, **kwargs):
        store[key] = value
        return True

    async def delete(key):
        store.pop(key, None)
        return 1

    async def incr(key):
        current = int(store.get(key, "0"))
        new_val = current + 1
        store[key] = str(new_val)
        return new_val

    r.get.side_effect = get
    r.set.side_effect = set
    r.delete.side_effect = delete
    r.incr.side_effect = incr
    return r, store


@pytest.fixture
def cb(mock_redis):
    redis, store = mock_redis
    breaker = CircuitBreaker(
        backend="test_backend",
        redis_client=redis,
    )
    breaker.failure_threshold = 3
    breaker.recovery_timeout = 5
    breaker.success_threshold = 2
    return breaker, store


@pytest.mark.asyncio
async def test_initial_state_is_closed(cb):
    breaker, store = cb
    assert await breaker.allow_request() is True


@pytest.mark.asyncio
async def test_opens_after_failure_threshold(cb):
    breaker, store = cb
    # Record failures up to threshold
    for _ in range(breaker.failure_threshold):
        assert await breaker.allow_request() is True
        await breaker.record_failure()

    # Circuit should now be OPEN
    assert await breaker.allow_request() is False


@pytest.mark.asyncio
async def test_success_resets_failure_count_in_closed_state(cb):
    breaker, store = cb
    # Trigger 2 failures (below threshold)
    await breaker.record_failure()
    await breaker.record_failure()
    # Recover with success
    await breaker.record_success()
    # Fail count should be reset — 3 more failures should open it
    for _ in range(breaker.failure_threshold):
        await breaker.record_failure()
    assert await breaker.allow_request() is False


@pytest.mark.asyncio
async def test_half_open_after_recovery_timeout(cb):
    breaker, store = cb
    # Force open
    for _ in range(breaker.failure_threshold):
        await breaker.record_failure()
    assert await breaker.allow_request() is False

    # Simulate elapsed recovery timeout
    past = str(time.time() - breaker.recovery_timeout - 1)
    store[breaker._k_opened_at] = past

    # Should now be allowed (HALF_OPEN probe)
    assert await breaker.allow_request() is True
    state = await breaker._get_state()
    assert state == CBState.HALF_OPEN


@pytest.mark.asyncio
async def test_half_open_to_closed_on_success(cb):
    breaker, store = cb
    # Force to HALF_OPEN
    for _ in range(breaker.failure_threshold):
        await breaker.record_failure()
    past = str(time.time() - breaker.recovery_timeout - 1)
    store[breaker._k_opened_at] = past
    await breaker.allow_request()  # transitions to HALF_OPEN

    # Record enough successes
    for _ in range(breaker.success_threshold):
        await breaker.record_success()

    state = await breaker._get_state()
    assert state == CBState.CLOSED


@pytest.mark.asyncio
async def test_half_open_to_open_on_failure(cb):
    breaker, store = cb
    # Force to HALF_OPEN
    for _ in range(breaker.failure_threshold):
        await breaker.record_failure()
    past = str(time.time() - breaker.recovery_timeout - 1)
    store[breaker._k_opened_at] = past
    await breaker.allow_request()  # HALF_OPEN

    await breaker.record_failure()  # probe fails
    state = await breaker._get_state()
    assert state == CBState.OPEN


@pytest.mark.asyncio
async def test_fail_open_when_redis_unavailable():
    breaker = CircuitBreaker(backend="no_redis")
    breaker._redis = None
    # Should always allow (fail-open)
    assert await breaker.allow_request() is True
    await breaker.record_failure()  # should not raise
    await breaker.record_success()  # should not raise


@pytest.mark.asyncio
async def test_get_status_structure(cb):
    breaker, _ = cb
    status = await breaker.get_status()
    assert "backend" in status
    assert "state" in status
    assert "fail_count" in status
    assert status["backend"] == "test_backend"
    assert status["state"] == "CLOSED"
