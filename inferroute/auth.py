import time
import logging
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as aioredis
from inferroute.config import settings
from inferroute.observability import RATE_LIMITED_TOTAL

logger = logging.getLogger("inferroute.auth")
security = HTTPBearer()

# Static API key mapping for MVP demo
API_KEYS = {
    "sk-inferroute-demo": "acme_corp",
    "sk-inferroute-dev": "internal_dev",
    settings.ADMIN_API_KEY: "admin"
}

# Redis Client Instance placeholder (initialized in main.py)
redis_client: Optional[aioredis.Redis] = None

def get_redis_client() -> Optional[aioredis.Redis]:
    global redis_client
    return redis_client

from inferroute.database import async_session
from inferroute.models import UserWallet
from sqlalchemy import select

async def check_wallet_balance(tenant_id: str) -> None:
    """
    Checks if a tenant has a positive wallet balance.
    If not, raises 402 Payment Required.
    Automatically creates a wallet with a trial balance if it doesn't exist.
    Fails open (logs a warning and continues) if Database is unavailable.
    """
    if tenant_id == "admin":
        return

    try:
        async with async_session() as session:
            result = await session.execute(
                select(UserWallet).where(UserWallet.tenant_id == tenant_id)
            )
            wallet = result.scalar_one_or_none()

            if wallet is None:
                wallet = UserWallet(tenant_id=tenant_id, balance_usd=5.0)
                session.add(wallet)
                await session.commit()
                logger.info(f"Created new trial wallet for tenant={tenant_id} with $5.00")
                return

            if wallet.balance_usd <= 0.0:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Payment Required: Wallet balance dry. Please recharge."
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database error during wallet balance check: {e}. Bypassing wallet check.")
        return

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verifies the provided API key and returns the associated tenant ID.
    Supports standard Bearer token header.
    """
    token = credentials.credentials
    if token in API_KEYS:
        tenant_id = API_KEYS[token]
        await check_wallet_balance(tenant_id)
        return tenant_id
    
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
