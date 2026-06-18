import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    ENV: str = Field(default="development", validation_alias="ENV")
    
    # Databases
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/inferroute",
        validation_alias="DATABASE_URL"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        validation_alias="REDIS_URL"
    )
    
    # API Keys & Third-party integrations
    OPENAI_API_KEY: str = Field(default="mock-key", validation_alias="OPENAI_API_KEY")
    VLLM_API_URL: str = Field(default="http://localhost:8000/v1", validation_alias="VLLM_API_URL")
    VLLM_API_KEY: str = Field(default="mock-vllm-key", validation_alias="VLLM_API_KEY")
    MOCK_VLLM: bool = Field(default=True, validation_alias="MOCK_VLLM")
    
    # Gatekeeper settings
    ADMIN_API_KEY: str = Field(default="admin-secret", validation_alias="ADMIN_API_KEY")
    DEFAULT_RATE_LIMIT_RPM: int = Field(default=60, validation_alias="DEFAULT_RATE_LIMIT_RPM")
    
    # Observability
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
