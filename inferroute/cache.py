"""
Enhanced caching layer for InferRoute.

Three tiers:
1. Exact cache  — SHA-256 hash of full request → cached response (TTL 5 min)
2. Prefix cache — longest common prefix of recent prompts → partial cache hint
                  Uses character-prefix matching on the concatenated message content.
3. Deduplication — coalesces identical in-flight requests via Redis pub/sub.
                   If an identical request is already being processed, wait for its
                   result rather than sending a duplicate to the provider.
"""
import asyncio
import hashlib
import json
import logging
from typing import Any, Optional

from inferroute.auth import get_redis_client
from inferroute.config import settings
from inferroute.observability import (
    CACHE_HIT_TOTAL,
    PREFIX_CACHE_HIT_TOTAL,
    DEDUP_HIT_TOTAL,
)

logger = logging.getLogger("inferroute.cache")

# TTL constants
EXACT_TTL_S = 300       # 5 minutes for exact hits
PREFIX_TTL_S = 120      # 2 minutes for prefix hints
DEDUP_LOCK_TTL_S = settings.CACHE_DEDUP_TIMEOUT_S  # max wait for in-flight result


class CacheLayer:

    # ── Key generation ────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_req(req: dict[str, Any]) -> dict[str, Any]:
        """Strip non-content keys that do not affect generation."""
        clean = req.copy()
        for key in ("routing", "metadata", "stream", "tenant_id"):
            clean.pop(key, None)
        return clean

    @staticmethod
    def _exact_key(req: dict[str, Any]) -> str:
        """SHA-256 of the normalized, sorted-key JSON."""
        clean = CacheLayer._normalize_req(req)
        canonical = json.dumps(clean, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"inferroute:cache:exact:{digest}"

    @staticmethod
    def _prompt_text(req: dict[str, Any]) -> str:
        """Concatenate all message contents into a single string for prefix matching."""
        return " ".join(m.get("content", "") for m in req.get("messages", []))

    @staticmethod
    def _prefix_key(prefix: str) -> str:
        """Key for storing prefix-cache hint."""
        prefix_trunc = prefix[: settings.CACHE_PREFIX_MAX_CHARS]
        digest = hashlib.sha256(prefix_trunc.encode("utf-8")).hexdigest()
        return f"inferroute:cache:prefix:{digest}"

    @staticmethod
    def _dedup_lock_key(req: dict[str, Any]) -> str:
        """Same as exact key but under a different namespace — used as a mutex."""
        clean = CacheLayer._normalize_req(req)
        canonical = json.dumps(clean, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"inferroute:dedup:lock:{digest}"

    @staticmethod
    def _dedup_channel(req: dict[str, Any]) -> str:
        """Redis pub/sub channel name for broadcasting the result."""
        clean = CacheLayer._normalize_req(req)
        canonical = json.dumps(clean, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"inferroute:dedup:channel:{digest}"

    # ── Exact cache ───────────────────────────────────────────────────────────

    async def lookup_exact(self, req: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Look up the full request in Redis. Returns parsed response dict or None."""
        client = get_redis_client()
        if client is None:
            return None
        try:
            key = self._exact_key(req)
            raw = await client.get(key)
            if raw:
                logger.info(f"[Cache] Exact HIT key={key[:24]}…")
                CACHE_HIT_TOTAL.labels(type="exact").inc()
                return json.loads(raw)
        except Exception as e:
            logger.error(f"[Cache] Exact lookup error: {e}")
        return None

    async def store_exact(
        self, req: dict[str, Any], resp: dict[str, Any], ttl_sec: int = EXACT_TTL_S
    ) -> None:
        """Store response in exact cache and update prefix hint."""
        client = get_redis_client()
        if client is None:
            return
        try:
            key = self._exact_key(req)
            await client.set(key, json.dumps(resp, ensure_ascii=False), ex=ttl_sec)
            logger.debug(f"[Cache] Stored exact key={key[:24]}… TTL={ttl_sec}s")

            # Also store a prefix hint pointing to this exact key
            prompt = self._prompt_text(req)
            if len(prompt) >= 32:
                prefix = prompt[: settings.CACHE_PREFIX_MAX_CHARS]
                prefix_key = self._prefix_key(prefix)
                await client.set(prefix_key, key, ex=PREFIX_TTL_S)
        except Exception as e:
            logger.error(f"[Cache] Store error: {e}")

    # ── Prefix cache ──────────────────────────────────────────────────────────

    async def lookup_prefix(self, req: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Check if a cached response exists for a request that shares a common
        prompt prefix with the current request.

        This is a hint-based approach: we check multiple prefix lengths
        (100%, 75%, 50% of CACHE_PREFIX_MAX_CHARS) to find the longest match.
        On a hit, the prefix entry points to the exact-cache key for retrieval.
        """
        client = get_redis_client()
        if client is None:
            return None

        prompt = self._prompt_text(req)
        if len(prompt) < 32:
            return None  # too short for meaningful prefix matching

        max_chars = settings.CACHE_PREFIX_MAX_CHARS
        # Try longest prefix first, then shorter prefixes
        prefix_lengths = [max_chars, int(max_chars * 0.75), int(max_chars * 0.5)]

        try:
            for plen in prefix_lengths:
                prefix = prompt[:plen]
                if len(prefix) < 16:
                    continue
                prefix_key = self._prefix_key(prefix)
                exact_key_ref = await client.get(prefix_key)
                if exact_key_ref:
                    raw = await client.get(exact_key_ref)
                    if raw:
                        logger.info(f"[Cache] Prefix HIT prefix_len={plen} chars")
                        PREFIX_CACHE_HIT_TOTAL.inc()
                        CACHE_HIT_TOTAL.labels(type="prefix").inc()
                        return json.loads(raw)
        except Exception as e:
            logger.error(f"[Cache] Prefix lookup error: {e}")

        return None

    # ── Request deduplication ─────────────────────────────────────────────────

    async def try_acquire_dedup_lock(self, req: dict[str, Any]) -> bool:
        """
        Attempt to acquire the in-flight dedup lock for this request.
        Returns True if this caller is the 'owner' (first to process the request).
        Returns False if another caller already owns it (should wait for result).
        """
        if not settings.CACHE_DEDUP_ENABLED:
            return True

        client = get_redis_client()
        if client is None:
            return True

        try:
            lock_key = self._dedup_lock_key(req)
            # NX = only set if not exists; returns True if set, None if already exists
            acquired = await client.set(lock_key, "1", nx=True, ex=DEDUP_LOCK_TTL_S)
            return bool(acquired)
        except Exception as e:
            logger.warning(f"[Cache] Dedup lock error: {e}")
            return True  # fail-open

    async def release_dedup_lock(self, req: dict[str, Any]) -> None:
        """Release the dedup lock after processing is complete."""
        client = get_redis_client()
        if client is None:
            return
        try:
            lock_key = self._dedup_lock_key(req)
            await client.delete(lock_key)
        except Exception as e:
            logger.warning(f"[Cache] Dedup lock release error: {e}")

    async def wait_for_dedup_result(
        self, req: dict[str, Any], backend: str = "unknown"
    ) -> Optional[dict[str, Any]]:
        """
        Wait for the owner request to complete and publish its result.
        Subscribes to the Redis pub/sub channel for this request hash.
        Returns the cached response or None on timeout.
        """
        if not settings.CACHE_DEDUP_ENABLED:
            return None

        client = get_redis_client()
        if client is None:
            return None

        channel = self._dedup_channel(req)
        logger.info(f"[Cache] Waiting for in-flight dedup result on channel={channel[:32]}…")

        try:
            # Use a new connection for pub/sub
            pubsub = client.pubsub()
            await pubsub.subscribe(channel)

            try:
                deadline = asyncio.get_event_loop().time() + DEDUP_LOCK_TTL_S
                while asyncio.get_event_loop().time() < deadline:
                    message = await asyncio.wait_for(pubsub.get_message(ignore_subscribe_messages=True), timeout=1.0)
                    if message and message.get("type") == "message":
                        data = message["data"]
                        if isinstance(data, bytes):
                            data = data.decode("utf-8")
                        if data == "__error__":
                            return None  # owner failed; caller should retry
                        DEDUP_HIT_TOTAL.labels(backend=backend).inc()
                        return json.loads(data)
            finally:
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()

        except Exception as e:
            logger.error(f"[Cache] Dedup wait error: {e}")

        return None

    async def publish_dedup_result(
        self, req: dict[str, Any], resp: Optional[dict[str, Any]]
    ) -> None:
        """
        Publish the response to all waiters on this request's pub/sub channel.
        Call this after the owner has successfully received a response.
        """
        client = get_redis_client()
        if client is None:
            return
        try:
            channel = self._dedup_channel(req)
            if resp is None:
                payload = "__error__"
            else:
                payload = json.dumps(resp, ensure_ascii=False)
            await client.publish(channel, payload)
            logger.debug(f"[Cache] Published dedup result to channel={channel[:32]}…")
        except Exception as e:
            logger.error(f"[Cache] Dedup publish error: {e}")
