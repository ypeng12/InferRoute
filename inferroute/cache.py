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
        
        # Pop routing, metadata, and stream
        routing = clean.pop("routing", {}) or {}
        metadata = clean.pop("metadata", {}) or {}
        clean.pop("stream", None)
        
        # Check if shared_cache=True is explicitly set to share cache across tenants
        shared_cache = False
        if isinstance(routing, dict):
            shared_cache = shared_cache or routing.get("shared_cache", False)
        if isinstance(metadata, dict):
            shared_cache = shared_cache or metadata.get("shared_cache", False)
            
        if not shared_cache:
            # Retain tenant_id to isolate cached entries by default
            pass
        else:
            clean.pop("tenant_id", None)
            
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
        """Store response in exact cache."""
        client = get_redis_client()
        if client is None:
            return
        try:
            key = self._exact_key(req)
            await client.set(key, json.dumps(resp, ensure_ascii=False), ex=ttl_sec)
            logger.debug(f"[Cache] Stored exact key={key[:24]}… TTL={ttl_sec}s")
        except Exception as e:
            logger.error(f"[Cache] Store error: {e}")

    # ── Prefix cache ──────────────────────────────────────────────────────────

    async def lookup_prefix(self, req: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Check if a cached response exists for a request that shares a common
        prompt prefix with the current request.

        DEPRECATED: Prefix cache is no longer used to directly return cached responses
        to avoid incorrect answers. We now use router_trie.py for warm KV cache routing affinity.
        """
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

    # ── Streaming request deduplication ───────────────────────────────────────

    def _stream_chunks_key(self, req: dict[str, Any]) -> str:
        clean = self._normalize_req(req)
        canonical = json.dumps(clean, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"inferroute:dedup:stream_chunks:{digest}"

    def _stream_channel(self, req: dict[str, Any]) -> str:
        clean = self._normalize_req(req)
        canonical = json.dumps(clean, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"inferroute:dedup:stream_channel:{digest}"

    async def push_stream_chunk(
        self, req: dict[str, Any], index: int, chunk: dict[str, Any]
    ) -> None:
        """Push a stream chunk to Redis list and publish to stream channel."""
        client = get_redis_client()
        if client is None:
            return
        try:
            list_key = self._stream_chunks_key(req)
            channel = self._stream_channel(req)
            payload = json.dumps({"index": index, "chunk": chunk}, ensure_ascii=False)
            
            # Pipe commands to ensure atomic push + publish
            pipe = client.pipeline()
            pipe.rpush(list_key, payload)
            pipe.expire(list_key, 120)  # 2 minutes TTL
            pipe.publish(channel, payload)
            await pipe.execute()
        except Exception as e:
            logger.error(f"[Cache] Push stream chunk error: {e}")

    async def publish_stream_end(self, req: dict[str, Any], final_index: int) -> None:
        """Signal the end of a stream to all waiters."""
        client = get_redis_client()
        if client is None:
            return
        try:
            list_key = self._stream_chunks_key(req)
            channel = self._stream_channel(req)
            payload = json.dumps({"index": "done", "final_index": final_index}, ensure_ascii=False)
            
            pipe = client.pipeline()
            pipe.rpush(list_key, payload)
            pipe.expire(list_key, 120)
            pipe.publish(channel, payload)
            await pipe.execute()
        except Exception as e:
            logger.error(f"[Cache] Publish stream end error: {e}")

    async def publish_stream_error(self, req: dict[str, Any], err_msg: str) -> None:
        """Signal a stream error to all waiters."""
        client = get_redis_client()
        if client is None:
            return
        try:
            list_key = self._stream_chunks_key(req)
            channel = self._stream_channel(req)
            payload = json.dumps({"index": "error", "error": err_msg}, ensure_ascii=False)
            
            pipe = client.pipeline()
            pipe.rpush(list_key, payload)
            pipe.expire(list_key, 120)
            pipe.publish(channel, payload)
            await pipe.execute()
        except Exception as e:
            logger.error(f"[Cache] Publish stream error: {e}")

    async def wait_for_stream_dedup(
        self, req: dict[str, Any]
    ) -> Any:
        """
        Wait for stream chunks from the owner request.
        Yields chunk dicts in real-time.
        """
        client = get_redis_client()
        if client is None:
            return

        channel = self._stream_channel(req)
        list_key = self._stream_chunks_key(req)
        
        # Subscribe to pubsub channel first to buffer incoming chunks
        pubsub = client.pubsub()
        await pubsub.subscribe(channel)
        
        try:
            # Read any chunks that have already been generated and cached in the list
            history = await client.lrange(list_key, 0, -1)
            history_chunks = []
            stream_finished = False
            stream_error = None
            
            for item in history:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                data = json.loads(item)
                idx = data.get("index")
                if idx == "done":
                    stream_finished = True
                elif idx == "error":
                    stream_error = data.get("error", "Stream error")
                else:
                    history_chunks.append((idx, data.get("chunk")))
            
            # Sort history chunks by index to ensure proper sequence
            history_chunks.sort(key=lambda x: x[0])
            for _, chunk in history_chunks:
                yield chunk
                
            if stream_finished:
                return
            if stream_error:
                raise ValueError(stream_error)
                
            # Consume live chunks from pub/sub channel
            next_expected_index = len(history_chunks)
            deadline = asyncio.get_event_loop().time() + DEDUP_LOCK_TTL_S
            
            while asyncio.get_event_loop().time() < deadline:
                try:
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=1.0
                    )
                    if message and message.get("type") == "message":
                        data_str = message["data"]
                        if isinstance(data_str, bytes):
                            data_str = data_str.decode("utf-8")
                        data = json.loads(data_str)
                        idx = data.get("index")
                        if idx == "done":
                            break
                        elif idx == "error":
                            raise ValueError(data.get("error", "Stream error"))
                        else:
                            if isinstance(idx, int) and idx >= next_expected_index:
                                yield data["chunk"]
                                next_expected_index = idx + 1
                except asyncio.TimeoutError:
                    # Safety check: if lock is gone, check if done was written to list
                    lock_exists = await client.exists(self._dedup_lock_key(req))
                    if not lock_exists:
                        history = await client.lrange(list_key, 0, -1)
                        for item in history:
                            if isinstance(item, bytes):
                                item = item.decode("utf-8")
                            d = json.loads(item)
                            if d.get("index") == "done":
                                return
                        break
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
