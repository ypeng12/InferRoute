# Build stage using official python 3.12 slim image
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

# Install system dependencies (optional, for potential sqlite compilation or git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project source code
COPY inferroute/ ./inferroute/
COPY docs/ ./docs/
COPY benchmarks/ ./benchmarks/
COPY external/ ./external/
COPY *.html ./
COPY assets/ ./assets/

# Expose default port (7860 for Hugging Face Spaces)
EXPOSE 7860

# Command to run uvicorn dynamically binding to PORT environment variable (for Render/Hugging Face compatibility)
CMD ["sh", "-c", "python -m uvicorn inferroute.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
