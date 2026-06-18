# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN groupadd -r inferroute && useradd -r -g inferroute -d /app -s /sbin/nologin inferroute

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only the application package (not edgeserve/ legacy dir)
COPY inferroute/ ./inferroute/

# Set ownership
RUN chown -R inferroute:inferroute /app

USER inferroute

# Expose the gateway port
EXPOSE 8080

# Health check via the /healthz endpoint
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "inferroute.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info"]
