# Stage 1: Build React frontend for Quant.ai
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY external/Quant.ai/frontend/package*.json ./
RUN npm install --legacy-peer-deps
COPY external/Quant.ai/frontend/ ./
RUN npm run build

# Stage 2: Python 3.12 high-performance inference gateway
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    DATABASE_URL=sqlite+aiosqlite:////app/inferroute.db \
    MOCK_OPENAI=true \
    MOCK_GEMINI=true \
    MOCK_VLLM=true \
    MOCK_OLLAMA=true

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY inferroute/ ./inferroute/
COPY docs/ ./docs/
COPY benchmarks/ ./benchmarks/
COPY external/ ./external/
COPY platform/ ./platform/
COPY quant/ ./quant/
COPY *.html ./
COPY assets/ ./assets/
COPY *.svg ./

# Copy compiled React frontend distribution bundle from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./external/Quant.ai/frontend/dist

# Expose default port (7860 for Hugging Face Spaces)
EXPOSE 7860

# Launch Uvicorn gateway
CMD ["sh", "-c", "python -m uvicorn inferroute.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
