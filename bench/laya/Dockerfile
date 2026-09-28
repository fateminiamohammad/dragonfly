# Laya (huggingface.co/convaiinnovations/laya, Apache-2.0) as a benchmark opponent, served the way its model card
# says: `pip install "laya[serve]"` then `LAYA_DEVICE=cuda LAYA_PRELOAD=1 laya-serve`. Pinned to laya 0.3.21.
#   docker compose -f docker-compose.local.yml --profile laya up -d laya
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/models/hf
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126 \
    && pip install --no-cache-dir "laya[serve]==0.3.21"
# ModernBERT on CUDA compiles kernels with triton, which needs a C compiler at runtime
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
ENV LAYA_DEVICE=cuda LAYA_PRELOAD=1 LAYA_MODELS=english,typed-decisions LAYA_HOST=0.0.0.0 LAYA_PORT=8010
EXPOSE 8010
CMD ["laya-serve"]
