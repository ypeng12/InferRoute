from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False)
    logical_model = Column(String(100), nullable=False)

    # ── Provider ──────────────────────────────────────────────────────────────
    provider = Column(String(50), nullable=True)  # openai | gemini | ollama | vllm | cache

    # ── Usage and costing ────────────────────────────────────────────────────
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)

    # ── Caching ───────────────────────────────────────────────────────────────
    cache_hit = Column(Boolean, default=False)
    cache_type = Column(String(50), nullable=True)   # exact, prefix
    prefix_cache_hit = Column(Boolean, default=False)
    dedup_hit = Column(Boolean, default=False)

    # ── Routing ───────────────────────────────────────────────────────────────
    primary_backend = Column(String(50), nullable=False)
    selected_backend = Column(String(50), nullable=False)
    fallback_count = Column(Integer, default=0)
    routing_policy = Column(String(50), nullable=True)   # latency | cost | reliability
    circuit_state = Column(String(20), nullable=True)    # CLOSED | OPEN | HALF_OPEN

    # ── Status ────────────────────────────────────────────────────────────────
    status = Column(String(50), default="completed")     # completed | failed | validation_failed | rate_limited
    error_message = Column(String(500), nullable=True)

    # ── SLO ───────────────────────────────────────────────────────────────────
    slo_met = Column(Boolean, default=True)
    slo_p95_target_ms = Column(Float, nullable=True)

    # ── Timing (milliseconds) ─────────────────────────────────────────────────
    timing_queue_ms = Column(Float, default=0.0)
    timing_ttft_ms = Column(Float, default=0.0)
    timing_latency_ms = Column(Float, default=0.0)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "model": self.model,
            "logical_model": self.logical_model,
            "provider": self.provider,
            "usage": {
                "input_tokens": self.prompt_tokens,
                "output_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "estimated_cost_usd": self.cost_usd,
            },
            "cache": {
                "hit": self.cache_hit,
                "type": self.cache_type,
                "prefix_cache_hit": self.prefix_cache_hit,
                "dedup_hit": self.dedup_hit,
            },
            "route": {
                "primary_backend": self.primary_backend,
                "selected_backend": self.selected_backend,
                "fallback_count": self.fallback_count,
                "policy": self.routing_policy,
                "circuit_state": self.circuit_state,
            },
            "slo": {
                "met": self.slo_met,
                "p95_target_ms": self.slo_p95_target_ms,
            },
            "status": self.status,
            "error_message": self.error_message,
            "timing": {
                "queue_ms": self.timing_queue_ms,
                "ttft_ms": self.timing_ttft_ms,
                "latency_ms": self.timing_latency_ms,
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class UserWallet(Base):
    __tablename__ = "user_wallets"

    tenant_id = Column(String(100), primary_key=True, index=True)
    balance_usd = Column(Float, default=5.0)  # Default trial balance
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )


class TransactionLedger(Base):
    __tablename__ = "transaction_ledger"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(String(100), nullable=False, index=True)
    amount_usd = Column(Float, nullable=False)  # positive for recharge, negative for deduction
    transaction_type = Column(String(50), nullable=False)  # "recharge" or "deduction"
    description = Column(String(200), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True
    )
