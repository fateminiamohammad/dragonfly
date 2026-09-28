# Kev (github.com/jaredpalmer/kev, Apache-2.0) as a benchmark opponent, served the way its README says
# (`python -m kev.serve`), with its own serving optimizations: bf16, CUDA graphs, fused Qwen3.5 kernels
# (flash-linear-attention, the version Kev pins). Built from Kev's repo at a pinned commit; nothing of Kev is vendored.
#   docker compose -f docker-compose.local.yml --profile kev up -d kev     # KEV_MODEL=jaredpalmer/kev-4b by default
FROM python:3.13-slim
ARG KEV_COMMIT=5920c5fe4ca8e0970ed4209ac2c9b8e18bea5109
ENV PYTHONUNBUFFERED=1 HF_HOME=/models/hf UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends git gcc g++ libc6-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
WORKDIR /kev
RUN git init -q . && git remote add origin https://github.com/jaredpalmer/kev.git \
    && git fetch -q --depth 1 origin ${KEV_COMMIT} && git checkout -q FETCH_HEAD
RUN uv sync --frozen --extra serve --no-dev \
    && uv pip install --python .venv/bin/python "flash-linear-attention==0.5.2"
EXPOSE 8009
CMD ["sh", "-c", ".venv/bin/python -m kev.serve --run ${KEV_MODEL:-jaredpalmer/kev-4b} --host 0.0.0.0 --port 8009"]
