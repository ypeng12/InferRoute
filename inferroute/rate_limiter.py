"""
Adaptive Concurrency Limiter for InferRoute.

Dynamically adjusts the maximum concurrent requests allowed through the gateway
based on queue size estimates derived from latency measurements (inspired by TCP Vegas).
Helps prevent upstream LLM/vLLM overload and maximizes gateway throughput.
"""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger("inferroute.rate_limiter")


class AdaptiveConcurrencyLimiter:
    """
    Vegas-style adaptive concurrency rate limiter.
    Adjusts dynamic concurrency limits based on min RTT (baseline latency)
    and current RTT (average latency of recent requests).
    """

    def __init__(
        self,
        initial_limit: int = 15,
        min_limit: int = 2,
        max_limit: int = 100,
        alpha: int = 2,  # queue threshold below which we increase limit
        beta: int = 5,   # queue threshold above which we decrease limit
    ):
        self.limit = initial_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        self.alpha = alpha
        self.beta = beta

        self.active_requests = 0
        self.min_latency_ms: Optional[float] = None
        self.latency_samples = []
        self.window_size = 30
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """
        Check if we can accept a new request.
        Returns True if request is allowed (increments active count), False if throttled.
        """
        async with self._lock:
            if self.active_requests < self.limit:
                self.active_requests += 1
                logger.debug(
                    f"[RateLimiter] Request allowed. Active={self.active_requests}/{self.limit}"
                )
                return True
            logger.warning(
                f"[RateLimiter] Throttled request. Active={self.active_requests} exceeds limit={self.limit}"
            )
            return False

    async def release(self, latency_ms: float) -> None:
        """
        Release a request slot and adjust the concurrency limit based on the request's latency.
        """
        async with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            
            # Skip limit updates if latency is invalid/too small
            if latency_ms <= 0:
                return

            # Update baseline min latency (min RTT)
            if self.min_latency_ms is None or latency_ms < self.min_latency_ms:
                self.min_latency_ms = latency_ms
                logger.info(f"[RateLimiter] New baseline latency established: {latency_ms:.1f}ms")

            # Add to sliding window of recent latencies
            self.latency_samples.append(latency_ms)
            if len(self.latency_samples) > self.window_size:
                self.latency_samples.pop(0)

            # Recalculate limit periodically or when we have enough samples
            if len(self.latency_samples) >= 5:
                avg_latency = sum(self.latency_samples) / len(self.latency_samples)
                min_lat = self.min_latency_ms
                
                # Estimate queue size (Vegas algorithm)
                # queue = limit * (1 - min_latency / avg_latency)
                expected_throughput = self.limit / min_lat
                actual_throughput = self.limit / avg_latency
                queue_est = int((expected_throughput - actual_throughput) * min_lat)

                old_limit = self.limit
                if queue_est < self.alpha:
                    # Low queue: additively increase capacity
                    self.limit = min(self.max_limit, self.limit + 1)
                elif queue_est > self.beta:
                    # High queue: multiplicatively decrease capacity to relieve overload
                    self.limit = max(self.min_limit, int(self.limit * 0.9))

                if self.limit != old_limit:
                    logger.info(
                        f"[RateLimiter] Adjusted concurrency limit: {old_limit} -> {self.limit} "
                        f"(avg_lat={avg_latency:.1f}ms, queue_est={queue_est})"
                    )
