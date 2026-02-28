# ── Stage 1: builder ─────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml VERSION README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /dist

# ── Stage 2: runtime ─────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="Guilherme Pinheiro"
LABEL description="PyAccelerate — High-performance Python acceleration engine"

# Install system deps for psutil
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    && rm -f /tmp/*.whl

# Verify installation
RUN pyaccelerate version

WORKDIR /app

ENTRYPOINT ["pyaccelerate"]
CMD ["info"]
