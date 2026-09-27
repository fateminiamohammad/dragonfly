"""Entry point: `dragonfly-serve` (or `python -m dragonfly.api.serve`).

Configuration is by environment (see docker/.env.example):
  DRAGONFLY_CHECKPOINT     checkpoint directory; or base:<encoder> / base-m:<decoder> for an untrained head
                           (default base:answerdotai/ModernBERT-base)
  DRAGONFLY_CHECKPOINT_M   optional second (tier M) checkpoint: enables the S -> M cascade
  DRAGONFLY_CASCADE_THRESHOLD  confidence below which S's answer is re-asked to M (default 0.8)
  DRAGONFLY_DEVICE         cuda | cpu (default: cuda when available)
  DRAGONFLY_CUDA_GRAPHS    1 (default) = replay tier S as CUDA graphs on GPU: ~5x lower latency; 0 = off
  DRAGONFLY_COMPILE        1 = torch.compile the backbones (slower start, faster steady state)
  DRAGONFLY_CACHE_SIZE     answers kept for repeated requests (default 4096, 0 = off)
  DRAGONFLY_API_KEYS       comma-separated bearer keys
  REDIS_URL                optional: API keys and usage shared with the backend
  DRAGONFLY_PLUGINS        comma-separated plugin names to load
  DRAGONFLY_MAX_BATCH      requests per forward pass (default 64)
  DRAGONFLY_HOST / DRAGONFLY_PORT   bind address (default 0.0.0.0:8000)
"""

from __future__ import annotations

import logging
import os

import torch
import uvicorn

from ..auth import KeyStore
from ..batching import Worker
from ..cache import AnswerCache
from ..engine import Cascade, Engine
from ..plugins import PluginHost
from .app import create_app

log = logging.getLogger("dragonfly.serve")


def load_engine(path: str, device: str | None) -> Engine:
    engine = Engine.load(path, device)
    if os.environ.get("DRAGONFLY_CUDA_GRAPHS", "1") == "1":
        engine.enable_cuda_graphs()
    if os.environ.get("DRAGONFLY_COMPILE") == "1":
        engine.model.backbone = torch.compile(engine.model.backbone, dynamic=True)
        log.info("torch.compile enabled for %s", path)
    return engine


def build_app():
    logging.basicConfig(level=os.environ.get("DRAGONFLY_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    device = os.environ.get("DRAGONFLY_DEVICE") or None
    engine = load_engine(os.environ.get("DRAGONFLY_CHECKPOINT", "base:answerdotai/ModernBERT-base"), device)
    if os.environ.get("DRAGONFLY_CHECKPOINT_M"):
        engine = Cascade(engine, load_engine(os.environ["DRAGONFLY_CHECKPOINT_M"], device),
                         float(os.environ.get("DRAGONFLY_CASCADE_THRESHOLD", "0.8")))
        log.info("cascade S -> M enabled at confidence < %.2f", engine.threshold)
    worker = Worker(engine, int(os.environ.get("DRAGONFLY_MAX_BATCH", "64")))

    redis = None
    if os.environ.get("REDIS_URL"):
        import redis.asyncio as aioredis

        redis = aioredis.from_url(os.environ["REDIS_URL"])
    static = [k.strip() for k in os.environ.get("DRAGONFLY_API_KEYS", "").split(",") if k.strip()]
    keys = KeyStore(static, redis)
    if not keys.enabled:
        log.warning("no API keys configured: /v1 is OPEN (fine locally, never in production)")
    cache = AnswerCache(int(os.environ.get("DRAGONFLY_CACHE_SIZE", "4096")))
    return create_app(worker, PluginHost.from_env(), keys=keys, cache=cache)


def main() -> None:
    uvicorn.run(build_app(), host=os.environ.get("DRAGONFLY_HOST", "0.0.0.0"),
                port=int(os.environ.get("DRAGONFLY_PORT", "8000")), log_level="info", access_log=False)


if __name__ == "__main__":
    main()
