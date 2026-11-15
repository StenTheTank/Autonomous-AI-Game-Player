FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BRASS_ENGINE_ROOT=/opt/brass \
    BRASS_WEBAPP_DIR=/opt/brass/webapp \
    BRASS_TRAJECTORY_PATH=/app/trajectory.jsonl \
    BRASS_EVAL_SEED=202611 \
    BRASS_EVAL_DIFFICULTY=easy

# System utilities needed by Chromium/Playwright driver, OpenCode agent setup, and CLI inspection
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        jq \
        libatomic1 \
        procps \
        tar \
        unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python scientific/RL stack (CPU PyTorch) + Playwright Chromium
RUN pip install --no-cache-dir numpy "playwright>=1.48.0" \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

RUN playwright install --with-deps chromium

# Pre-install Node.js 22 + OpenCode CLI (`opencode-ai`) so `harbor run -a opencode`
# skips 6-minute per-trial runtime setup and starts playing in < 1 second.
RUN apt-get update && apt-get install -y --no-install-recommends coreutils xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && ARCH="$(dpkg --print-architecture)" \
    && NODE_ARCH="$([ "$ARCH" = "arm64" ] && echo "arm64" || echo "x64")" \
    && curl -fsSL "https://nodejs.org/dist/v22.14.0/node-v22.14.0-linux-${NODE_ARCH}.tar.xz" \
        | tar -xJ -C /usr/local --strip-components=1 \
    && npm install -g opencode-ai@latest \
    && opencode --version

# Install offline Brass: Birmingham simulator + engine + CLI
RUN mkdir -p /opt/brass /app /workspace
COPY webapp /opt/brass/webapp
COPY brass_engine /opt/brass/brass_engine
COPY brass_cli.py /opt/brass/brass_cli.py

RUN cp /opt/brass/brass_cli.py /usr/local/bin/brass \
    && sed -i 's/\r$//' /usr/local/bin/brass /opt/brass/brass_cli.py \
    && chmod 0755 /usr/local/bin/brass

ENV BRASS_ENFORCE_SINGLE_BOOT=1

WORKDIR /workspace
