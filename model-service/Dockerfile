# Dragonfly model-service. Stages: base -> dev (hot reload) -> runtime (non-root, production).
# TORCH_INDEX_URL selects the PyTorch build: CUDA 12.6 by default, https://download.pytorch.org/whl/cpu for CI / CPU hosts.
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf
WORKDIR /app
# torch.compile / triton (used by ModernBERT on CUDA and by DRAGONFLY_COMPILE) needs a C compiler at runtime
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
# torch first and alone: the biggest layer, and it changes least often
RUN pip install torch --index-url ${TORCH_INDEX_URL}
COPY pyproject.toml ./
RUN mkdir -p src/dragonfly && touch src/dragonfly/__init__.py \
    && pip install ".[train,grpc]" && pip uninstall -y dragonfly-model

FROM base AS dev
COPY . .
RUN pip install -e ".[train,grpc,dev]"
EXPOSE 8000
CMD ["uvicorn", "dragonfly.api.serve:build_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "src"]

FROM base AS runtime
COPY src ./src
RUN pip install --no-deps . \
    && groupadd --system dragonfly && useradd --system --gid dragonfly --create-home dragonfly \
    && mkdir -p /models/hf /runs && chown -R dragonfly:dragonfly /models /runs
USER dragonfly
EXPOSE 8000
CMD ["dragonfly-serve"]
