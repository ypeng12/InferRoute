# 🛡️ InferRoute Failure Injection & High-Availability Policies

This document details how **InferRoute** handles infrastructure failures (Redis, PostgreSQL, or LLM providers) to guarantee system reliability and maintain SLAs.

---

## 💥 Summary of Fault-Tolerance Strategies

| Component | Failure Mode | Impact Without Gateway | Gateway Mitigation Policy | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Redis** | Server crashes or network cut | Complete outage / API crash | **Fail-Open**: Bypasses caches & deduplication locks; routes requests directly to models. | Verified |
| **PostgreSQL** | Storage full or connection refused | Request crashes / Auth failure | **Fail-Open**: Bypasses balance checks and background audit logs; clients continue chatting. | Verified |
| **LLM Provider** | API timeout or HTTP 500 error | Bad user experience / app error | **Circuit Breaker & Fallback**: Reroutes subsequent queries to backup cloud models immediately. | Verified |

---

## 🔍 Detailed Fault Mitigation Logic

### 1. Redis Down (Cache & Limiter Bypass)
When the Redis cache server goes offline:
* **Connection Handling**: The connection is managed lazily by `redis.asyncio`. When commands like `GET`, `SET`, or `INCR` are run, they raise a connection error.
* **Code Implementation**:
  All Redis lookups in [cache.py](file:///c:/Users/pengy/OneDrive/Desktop/InferRoute/inferroute/cache.py) and rate limiters in [rate_limiter.py](file:///c:/Users/pengy/OneDrive/Desktop/InferRoute/inferroute/rate_limiter.py) are wrapped in `try...except Exception` blocks:
  ```python
  try:
      # Redis operations
      acquired = await client.set(lock_key, "1", nx=True, ex=30)
  except Exception as e:
      logger.warning(f"[Cache] Dedup lock error: {e}")
      return True  # Fail-open: proceed with standard model call
  ```
* **Impact**: Deduplication and caching are temporarily disabled, but requests continue to resolve successfully.

---

### 2. PostgreSQL Down (Auth & Audit Resiliency)
When the relational audit database is unreachable:
* **Wallet Balance Lookup**: In [auth.py](file:///c:/Users/pengy/OneDrive/Desktop/InferRoute/inferroute/auth.py), the balance check query is wrapped in a fail-open block.
  ```python
  try:
      async with async_session() as session:
          # Query UserWallet...
  except Exception as e:
      logger.error(f"Database error during wallet balance check: {e}. Bypassing wallet check.")
      return  # Fail-open: bypass credit restrictions during outages
  ```
* **Background Logging**: In [main.py](file:///c:/Users/pengy/OneDrive/Desktop/InferRoute/inferroute/main.py), writing requests logs and deducting balances runs asynchronously in a FastAPI background task:
  ```python
  try:
      # Write RequestLog & deduct balance
      session.add(log_entry)
      await session.commit()
  except Exception as e:
      logger.error(f"Failed to write request log & balance deduction: {e}")
  ```
  If the commit fails, it logs an error to stderr/logging stack but **does not crash the HTTP stream response**, protecting client connections.

---

### 3. Upstream LLM Timeout / Outage (Circuit Breakers)
If a primary model provider (e.g. local Ollama) runs out of memory or begins returning HTTP 500 errors:
* **Circuit Breaker State Machine**:
  * Tracked in [circuit_breaker.py](file:///c:/Users/pengy/OneDrive/Desktop/InferRoute/inferroute/circuit_breaker.py).
  * If the failure count exceeds `CB_FAILURE_THRESHOLD` (default: 5), the circuit breaker transitions from `CLOSED` to `OPEN`.
  * All subsequent requests are blocked from hitting the failed primary node and are instantly rerouted to the secondary cloud backend (e.g. OpenAI).
* **Self-Healing (Half-Open)**:
  * After `CB_RECOVERY_TIMEOUT_S` (default: 30 seconds), the circuit breaker enters the `HALF-OPEN` state.
  * It permits a limited number of trial requests to test backend health. If these requests succeed, the breaker transitions back to `CLOSED` (healed); if they fail, it transitions back to `OPEN`.
* **Testing Outages**:
  To verify this failover, run the chaos tests:
  ```bash
  python -m pytest tests/chaos_test.py
  ```
