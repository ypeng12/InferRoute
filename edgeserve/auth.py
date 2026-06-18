import time
import logging
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as aioredis
from edgeserve.config import settings
from edgeserve.observability import RATE_LIMITED_TOTAL

logger = logging.getLogger("edgeserve.auth")
security = HTTPBearer()

# Static API key mapping for MVP demo
API_KEYS = {
    "sk-edgeserve-demo": "acme_corp",
    "sk-edgeserve-dev": "internal_dev",
    settings.ADMIN_API_KEY: "admin"
}

# Redis Client Instance placeholder (initialized in main.py)
redis_client: Optional[aioredis.Redis] = None

def get_redis_client() -> Optional[aioredis.Redis]:
    global redis_client
    return redis_client

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verifies the provided API key and returns the associated tenant ID.
    Supports standard Bearer token header.
    """
    token = credentials.credentials
    if token in API_KEYS:
        return API_KEYS[token]
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key"
    )

async def check_rate_limit(tenant_id: str) -> None:
    """
    Checks request rate limits for a given tenant using Redis.
    Limits requests per minute (RPM).
    Fails open (logs a warning and continues) if Redis is unavailable.
    """
    client = get_redis_client()
    if client is None:
        logger.warning("Redis is not configured. Skipping rate limit checks.")
        return
        
    try:
        # Determine tenant limit
        limit = settings.DEFAULT_RATE_LIMIT_RPM
        if tenant_id == "admin":
            limit = 999999 # effectively unlimited
            
        current_minute = int(time.time() / 60)
        key = f"rate_limit:{tenant_id}:{current_minute}"
        
        # Increment request count for current minute window
        requests = await client.incr(key)
        if requests == 1:
            await client.expire(key, 60)
            
        if requests > limit:
            RATE_LIMITED_TOTAL.labels(scope="tenant").inc()
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum allowed is {limit} RPM."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Redis rate limiter exception: {e}. Bypassing rate limiting.")
        return
