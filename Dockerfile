# German Legal QA – Docker image for the demo
# Multi-stage build: install deps → copy source → run Gradio

FROM python:3.11-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (layer cache)
COPY pyproject.toml ./
RUN uv sync --no-dev --no-editable

# Copy source code
COPY src/ src/
COPY demo/ demo/
COPY configs/ configs/

# Copy pre-built data (if available – typically mounted as volume)
# COPY data/processed/ data/processed/

# Expose Gradio port
EXPOSE 7860

# Environment
ENV GLQA_DEVICE=cpu
ENV GRADIO_SERVER_NAME=0.0.0.0

# Run demo
CMD ["uv", "run", "python", "-m", "demo.app"]
