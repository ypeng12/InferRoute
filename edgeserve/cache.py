import hashlib
import json
import logging
from typing import Optional, Any
from edgeserve.auth import get_redis_client
from edgeserve.observability import CACHE_HIT_TOTAL

logger = logging.getLogger("edgeserve.cache")

class CacheLayer:
    @staticmethod
    def _exact_key(req: dict[str, Any]) -> str:
        """
        Creates a deterministic hash of the request dictionary to serve as cache key.
        Excludes optional keys like metadata, routing, and stream which do not affect content generation.
        """
        # Create a copy to avoid mutating the original request
        clean_req = req.copy()
        clean_req.pop("routing", None)
        clean_req.pop("metadata", None)
        clean_req.pop("stream", None) # Caching is content-based; we store & return non-stream response representation
        
        # Canonicalize JSON
        canonical = json.dumps(clean_req, sort_keys=True, ensure_ascii=False)
        return "edgeserve:cache:exact:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def lookup_exact(self, req: dict[str, Any]) -> Optional[dict[str, Any]]:
        """
        Looks up an exact request in Redis cache. Returns the parsed response dict if hit, else None.
        """
        client = get_redis_client()
        if client is None:
            return None
            
        try:
            key = self._exact_key(req)
            raw = await client.get(key)
            if raw:
                logger.info(f"Exact cache HIT for request hash: {key}")
                CACHE_HIT_TOTAL.labels(type="exact").inc()
                return json.loads(raw)
        except Exception as e:
            logger.error(f"Redis cache lookup error: {e}")
            
        return None

    async def store_exact(self, req: dict[str, Any], resp: dict[str, Any], ttl_sec: int = 300) -> None:
        """
        Stores a response dictionary in Redis cache with a TTL (Time-To-Live).
        """
        client = get_redis_client()
        if client is None:
            return
            
        try:
            key = self._exact_key(req)
            await client.set(key, json.dumps(resp, ensure_ascii=False), ex=ttl_sec)
            logger.info(f"Stored response in cache with key: {key}, TTL: {ttl_sec}s")
        except Exception as e:
            logger.error(f"Redis cache store error: {e}")
