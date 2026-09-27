# Dragonfly perception-service. Stages: base -> dev (hot reload) -> runtime (non-root).
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
# ffmpeg: audio formats libsndfile can't read (m4a, opus, webm). gcc: triton / torch.compile.
# libgl/libglib: OpenCV, used by RapidOCR for image processing.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg gcc libc6-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch --index-url ${TORCH_INDEX_URL}
COPY pyproject.toml ./
RUN mkdir -p src/perception && touch src/perception/__init__.py \
    && pip install . && pip uninstall -y dragonfly-perception

FROM base AS dev
COPY . .
RUN pip install -e ".[dev]"
EXPOSE 8001
CMD ["uvicorn", "perception.api:build_app", "--factory", "--host", "0.0.0.0", "--port", "8001", "--reload", "--reload-dir", "src"]

FROM base AS runtime
COPY src ./src
RUN pip install --no-deps . \
    && groupadd --system dragonfly && useradd --system --gid dragonfly --create-home dragonfly \
    && mkdir -p /models/hf && chown -R dragonfly:dragonfly /models /usr/local/lib/python3.12/site-packages/rapidocr
USER dragonfly
EXPOSE 8001
CMD ["dragonfly-perception"]
