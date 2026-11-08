# syntax=docker/dockerfile:1

# ---- builder: compiles the Rust core into the package ----
FROM python:3.15-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:$PATH" UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /application
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev || uv sync --no-dev

# ---- runtime ----
FROM python:3.15-slim AS runtime
RUN useradd --create-home --uid 1000 application
WORKDIR /application
COPY --from=builder --chown=application:application /application /application
ENV PATH="/application/.venv/bin:$PATH" PYTHONUNBUFFERED=1 DLP_HOST=0.0.0.0 DLP_PORT=8080
USER application
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')" || exit 1
CMD ["python", "-m", "llm_dlp_guard", "serve"]
