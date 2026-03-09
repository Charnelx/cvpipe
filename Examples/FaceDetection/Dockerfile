FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
WORKDIR /app

# Install uv globally from Astral
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies needed by opencv-python-headless & python3.13 via uv
RUN apt-get update && apt-get install -y libglib2.0-0 curl git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --python 3.13

COPY detector/ ./detector/
COPY scripts/ ./scripts/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Expose API port
EXPOSE 8000

# Download model then start uvicorn
CMD ["sh", "-c", "python scripts/download_model.py && uvicorn detector.app:app --host 0.0.0.0 --port 8000", "--log-level", "debug"]
