"""
Distributed Prefix Hash Trie routing for InferRoute.

Maintains prompt prefix caches in Redis using SHA-256 hashes of varying prefix lengths
(e.g., 128, 256, 512, 1024, 2048, 4096 characters). This allows longest-prefix cache
affinity matching across distributed backend hosts.
"""
import hashlib
import logging
from typing import List, Optional
from inferroute.auth import get_redis_client

logger = logging.getLogger("inferroute.router_trie")

# Define standard prefix lengths to index
PREFIX_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
TRIE_KEY_TTL_S = 600  # 10 minutes cache affinity lifetime


class PrefixTrieRouter:
    """
    Manages distributed cache affinity mappings.
    Registers which backend host processed which prompt prefixes,
    and retrieves affinity hosts by finding the longest matching prefix hash.
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client

    def _get_client(self):
        return self._redis if self._redis is not None else get_redis_client()

    def _prefix_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_key(self, prefix_hash: str) -> str:
        return f"inferroute:trie:{prefix_hash}"

    async def register_host_prefix(self, host: str, prompt_text: str) -> None:
        """
        Register a host as having the KV cache for prefixes of prompt_text.
        Computes hashes for all standard prefix lengths <= len(prompt_text).
        """
        client = self._get_client()
        if client is None:
            return

        try:
            pipe = client.pipeline()
            registered_any = False
            for length in PREFIX_LENGTHS:
                if len(prompt_text) >= length:
                    prefix = prompt_text[:length]
                    p_hash = self._prefix_hash(prefix)
                    key = self._get_key(p_hash)
                    pipe.sadd(key, host)
                    pipe.expire(key, TRIE_KEY_TTL_S)
                    registered_any = True
            
            if registered_any:
                await pipe.execute()
                logger.debug(f"[Trie] Registered prefixes for host={host}")
        except Exception as e:
            logger.error(f"[Trie] Failed to register prefixes for {host}: {e}")

    async def get_affinity_hosts(self, prompt_text: str) -> List[str]:
        """
        Find backend hosts that have processed the longest matching prefix of prompt_text.
        Scans from longest prefix length to shortest.
        """
        client = self._get_client()
        if client is None:
            return []

        # Try longest prefixes first
        for length in reversed(PREFIX_LENGTHS):
            if len(prompt_text) >= length:
                prefix = prompt_text[:length]
                p_hash = self._prefix_hash(prefix)
                key = self._get_key(p_hash)
                try:
                    hosts = await client.smembers(key)
                    if hosts:
                        hosts_list = [h.decode("utf-8") if isinstance(h, bytes) else str(h) for h in hosts]
                        logger.info(f"[Trie] Cache affinity MATCH at prefix_len={length} -> hosts={hosts_list}")
                        return hosts_list
                except Exception as e:
                    logger.error(f"[Trie] Match lookup error for length={length}: {e}")
                    
        return []
