import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    ENV: str = Field(default="development", validation_alias="ENV")

    # ── Databases ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/inferroute",
        validation_alias="DATABASE_URL"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        validation_alias="REDIS_URL"
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(default="mock-key", validation_alias="OPENAI_API_KEY")

    # ── Google Gemini ─────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = Field(default="mock-gemini-key", validation_alias="GEMINI_API_KEY")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash", validation_alias="GEMINI_MODEL")
    MOCK_GEMINI: bool = Field(default=True, validation_alias="MOCK_GEMINI")

    # ── vLLM ──────────────────────────────────────────────────────────────────
    VLLM_API_URL: str = Field(default="http://localhost:8000/v1", validation_alias="VLLM_API_URL")
    VLLM_API_KEY: str = Field(default="mock-vllm-key", validation_alias="VLLM_API_KEY")
    MOCK_VLLM: bool = Field(default=True, validation_alias="MOCK_VLLM")

    # ── Ollama ────────────────────────────────────────────────────────────────
    OLLAMA_API_URL: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_API_URL")
    OLLAMA_MODEL: str = Field(default="llama3", validation_alias="OLLAMA_MODEL")
    MOCK_OLLAMA: bool = Field(default=True, validation_alias="MOCK_OLLAMA")

    # ── Gateway Administration ────────────────────────────────────────────────
    ADMIN_API_KEY: str = Field(default="admin-secret", validation_alias="ADMIN_API_KEY")
    DEFAULT_RATE_LIMIT_RPM: int = Field(default=60, validation_alias="DEFAULT_RATE_LIMIT_RPM")

    # ── SLO Targets (milliseconds) ────────────────────────────────────────────
    SLO_P50_MS: float = Field(default=500.0, validation_alias="SLO_P50_MS")
    SLO_P95_MS: float = Field(default=2000.0, validation_alias="SLO_P95_MS")
    SLO_P99_MS: float = Field(default=5000.0, validation_alias="SLO_P99_MS")

    # ── Circuit Breaker ───────────────────────────────────────────────────────
    CB_FAILURE_THRESHOLD: int = Field(default=5, validation_alias="CB_FAILURE_THRESHOLD")
    CB_RECOVERY_TIMEOUT_S: int = Field(default=30, validation_alias="CB_RECOVERY_TIMEOUT_S")
    CB_SUCCESS_THRESHOLD: int = Field(default=2, validation_alias="CB_SUCCESS_THRESHOLD")

    # ── Caching ───────────────────────────────────────────────────────────────
    CACHE_MAX_SIZE_MB: int = Field(default=512, validation_alias="CACHE_MAX_SIZE_MB")
    CACHE_PREFIX_MAX_CHARS: int = Field(default=256, validation_alias="CACHE_PREFIX_MAX_CHARS")
    CACHE_DEDUP_ENABLED: bool = Field(default=True, validation_alias="CACHE_DEDUP_ENABLED")
    CACHE_DEDUP_TIMEOUT_S: int = Field(default=30, validation_alias="CACHE_DEDUP_TIMEOUT_S")

    # ── Observability ─────────────────────────────────────────────────────────
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(
        default="http://localhost:4317",
        validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    SERVICE_NAME: str = "inferroute-gateway"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
