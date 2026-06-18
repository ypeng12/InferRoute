import logging
from typing import Optional
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from inferroute.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inferroute.observability")

# ── PROMETHEUS METRICS ────────────────────────────────────────────────────────

# -- Counters --
REQUESTS_TOTAL = Counter(
    "inferroute_request_total",
    "Total requests routed",
    ["tenant", "model", "backend", "status"]
)

CACHE_HIT_TOTAL = Counter(
    "inferroute_cache_hit_total",
    "Total cache hit count",
    ["type"]  # exact, prefix
)

PROVIDER_COST_USD_TOTAL = Counter(
    "inferroute_provider_cost_usd_total",
    "Total accumulated token cost in USD",
    ["backend", "tenant"]
)

VALIDATION_FAIL_TOTAL = Counter(
    "inferroute_validation_fail_total",
    "Total output validation failures",
    ["reason"]
)

ROUTING_DECISION_TOTAL = Counter(
    "inferroute_routing_decision_total",
    "Total routing decisions made",
    ["policy", "backend"]
)

FALLBACK_TOTAL = Counter(
    "inferroute_fallback_total",
    "Total fallback occurrences",
    ["from_backend", "to_backend", "reason"]
)

RATE_LIMITED_TOTAL = Counter(
    "inferroute_rate_limited_total",
    "Total requests rate limited",
    ["scope"]  # tenant, global
)

SLO_VIOLATION_TOTAL = Counter(
    "inferroute_slo_violation_total",
    "Total SLO violations (requests exceeding target latency)",
    ["backend", "percentile"]  # percentile: p50, p95, p99
)

CIRCUIT_BREAKER_TRIP_TOTAL = Counter(
    "inferroute_circuit_breaker_trip_total",
    "Total circuit breaker trips (CLOSED→OPEN transitions)",
    ["backend"]
)

DEDUP_HIT_TOTAL = Counter(
    "inferroute_dedup_hit_total",
    "Total requests coalesced via in-flight deduplication",
    ["backend"]
)

PREFIX_CACHE_HIT_TOTAL = Counter(
    "inferroute_prefix_cache_hit_total",
    "Total prefix cache hits",
    []
)

# -- Histograms --
REQUEST_LATENCY = Histogram(
    "inferroute_request_latency_seconds",
    "End-to-end request latency in seconds",
    ["tenant", "model", "backend"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
)

TTFT_LATENCY = Histogram(
    "inferroute_ttft_seconds",
    "Time to first token (TTFT) in seconds",
    ["tenant", "model", "backend"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

INTER_TOKEN_LATENCY = Histogram(
    "inferroute_inter_token_latency_seconds",
    "Inter-token generation latency in seconds",
    ["tenant", "model", "backend"],
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
)

# -- Gauges --
QUEUE_DEPTH = Gauge(
    "inferroute_queue_depth",
    "Current active request count per backend (proxy for queue depth)",
    ["backend"]
)

CIRCUIT_BREAKER_STATE = Gauge(
    "inferroute_circuit_breaker_state",
    "Circuit breaker state per backend (0=CLOSED, 1=OPEN, 2=HALF_OPEN)",
    ["backend"]
)

# Sliding-window percentile estimates (updated by router after each request)
P50_LATENCY_MS = Gauge("inferroute_p50_latency_ms", "Estimated p50 latency (ms)", ["backend"])
P95_LATENCY_MS = Gauge("inferroute_p95_latency_ms", "Estimated p95 latency (ms)", ["backend"])
P99_LATENCY_MS = Gauge("inferroute_p99_latency_ms", "Estimated p99 latency (ms)", ["backend"])

# ── OPENTELEMETRY SETUP ───────────────────────────────────────────────────────
tracer = trace.get_tracer("inferroute")


def setup_observability(app: FastAPI) -> None:
    """Initialize OpenTelemetry and Prometheus instrumentation."""
    try:
        resource = Resource.create(attributes={
            "service.name": settings.SERVICE_NAME,
            "service.environment": settings.ENV
        })
        provider = TracerProvider(resource=resource)
        otlp_exporter = OTLPSpanExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=True
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(span_processor)
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        logger.info("OpenTelemetry instrumentation completed successfully.")
    except Exception as e:
        logger.warning(
            f"Failed to initialize OpenTelemetry tracing: {e}. Running without distributed tracing."
        )


def get_metrics_response() -> tuple[bytes, str]:
    """Generate Prometheus scrape-compatible response."""
    return generate_latest(), CONTENT_TYPE_LATEST
