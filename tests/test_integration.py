import pytest
import os
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, delete
import redis.asyncio as aioredis

from inferroute.config import settings
from inferroute.models import Base, UserWallet, RequestLog
from inferroute.database import init_db

@pytest.mark.asyncio
async def test_postgres_redis_integration():
    """
    Integration test validating connection, schemas, and basic queries against 
    real Redis and Postgres. Dynamically skips if services are unreachable.
    """
    # 1. Test Redis Connectivity
    redis_url = settings.REDIS_URL
    print(f"Testing Redis connection at {redis_url}...")
    try:
        redis_client = aioredis.from_url(redis_url)
        await redis_client.ping()
        print("Redis is reachable.")
    except Exception as e:
        pytest.skip(f"Redis is unreachable: {e}")
        
    # 2. Test PostgreSQL Connectivity
    db_url = settings.DATABASE_URL
    print(f"Testing database connection at {db_url}...")
    if "postgresql" not in db_url:
        await redis_client.close()
        pytest.skip("DATABASE_URL is not PostgreSQL, skipping integration test.")
        
    try:
        engine = create_async_engine(db_url, echo=False)
        async with engine.begin() as conn:
            # Check if we can run simple query
            await conn.execute(select(1))
        print("PostgreSQL is reachable.")
    except Exception as e:
        await redis_client.close()
        pytest.skip(f"PostgreSQL is unreachable: {e}")

    # 3. Perform Schema initialization
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    async_session = sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    test_tenant = "integration-test-tenant"
    
    # 4. Perform database operations (Upsert User Wallet)
    async with async_session() as session:
        try:
            # Delete old request logs if any
            await session.execute(
                delete(RequestLog).where(RequestLog.tenant_id == test_tenant)
            )
            
            # Upsert user wallet with $10.0 balance
            wallet = await session.get(UserWallet, test_tenant)
            if wallet:
                wallet.balance_usd = 10.0
            else:
                wallet = UserWallet(tenant_id=test_tenant, balance_usd=10.0)
                session.add(wallet)
            await session.commit()
        except Exception as e:
            await session.rollback()
            await redis_client.close()
            await engine.dispose()
            raise e
            
    # Verify database record exists
    async with async_session() as session:
        wallet = await session.get(UserWallet, test_tenant)
        assert wallet is not None
        assert wallet.balance_usd == 10.0
        
    # 5. Test Redis Caching & locks operations
    cache_key = f"inferroute:cache:exact:integration-test-key"
    try:
        # Store a value in Redis
        await redis_client.set(cache_key, "integration-test-value", ex=60)
        val = await redis_client.get(cache_key)
        assert val is not None
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        assert val == "integration-test-value"
    finally:
        # Clean up Redis
        await redis_client.delete(cache_key)
        await redis_client.close()
        await engine.dispose()
