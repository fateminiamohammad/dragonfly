"""Entry point: `dragonfly-serve` (or `python -m dragonfly.api.serve`).

Configuration is by environment (see docker/.env.example):
  DRAGONFLY_CHECKPOINT     checkpoint directory; or base:<encoder> / base-m:<decoder> for an untrained head
                           (default base:answerdotai/ModernBERT-base)
  DRAGONFLY_CHECKPOINT_M   optional second (tier M) checkpoint: enables the S -> M cascade
  DRAGONFLY_CASCADE_THRESHOLD  confidence below which S's answer is re-asked to M (default 0.8)
  DRAGONFLY_DEVICE         cuda | cpu (default: cuda when available)
  DRAGONFLY_CUDA_GRAPHS    1 (default) = replay forward passes as CUDA graphs on GPU: ~4-5x lower latency; 0 = off
  DRAGONFLY_WARMUP         1 (default) = capture common graph shapes at startup (no slow first requests)
  DRAGONFLY_STATE_CACHE    tier M: documents whose KV cache is kept for reuse (default 32, 0 = off)
  DRAGONFLY_COMPILE        1 = torch.compile the backbones (slower start, faster steady state)
  DRAGONFLY_CACHE_SIZE     answers kept for repeated requests (default 4096, 0 = off)
  DRAGONFLY_API_KEYS       comma-separated bearer keys
  REDIS_URL                optional: API keys and usage shared with the backend
  PERCEPTION_URL           optional: perception-service base URL; enables images and audio in the state
  DRAGONFLY_PLUGINS        comma-separated plugin names to load
  DRAGONFLY_MAX_BATCH      requests per forward pass (default 64)
  DRAGONFLY_SPECIALISTS    optional folder of specialists (docs/SWARM.md); requests pick one with "model"
  DRAGONFLY_SWARM_MAX_LOADED   tier S specialists kept in VRAM at once (default 4, least recently used unloaded)
  DRAGONFLY_ROUTE_THRESHOLD    "model": "auto" uses a specialist only above this router confidence (default 0.5)
  DRAGONFLY_HOST / DRAGONFLY_PORT   bind address (default 0.0.0.0:8000)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import torch
import uvicorn

from ..auth import KeyStore
from ..batching import Worker
from ..cache import AnswerCache
from ..engine import Cascade, Engine
from ..media import MediaResolver
from ..plugins import PluginHost
from ..swarm import AdapterBank, Swarm, read_card
from .app import create_app

log = logging.getLogger("dragonfly.serve")
RELOAD_CHANNEL = "dragonfly:specialists"


def load_engine(path: str, device: str | None, merge: bool = True) -> Engine:
    engine = Engine.load(path, device, merge=merge)
    if os.environ.get("DRAGONFLY_CUDA_GRAPHS", "1") == "1":
        engine.enable_cuda_graphs()
        if os.environ.get("DRAGONFLY_WARMUP", "1") == "1":
            log.info("captured %d CUDA graphs at startup for %s", engine.warmup(), path)
    engine.enable_state_cache(int(os.environ.get("DRAGONFLY_STATE_CACHE", "32")))
    if os.environ.get("DRAGONFLY_COMPILE") == "1":
        engine.model.backbone = torch.compile(engine.model.backbone, dynamic=True)
        log.info("torch.compile enabled for %s", path)
    return engine


def has_m_specialists(root: Path) -> bool:
    if not root.is_dir():
        return False
    for folder in root.iterdir():
        try:
            card = read_card(folder)
        except (OSError, ValueError, KeyError):
            continue
        if card is not None and card["tier"] == "M":
            return True
    return False


def listen_for_reloads(url: str, swarm: Swarm) -> None:
    """The backend publishes on dragonfly:specialists when a specialist is added or removed (all replicas reload)."""
    import redis

    def run():
        while True:
            try:
                pubsub = redis.Redis.from_url(url).pubsub(ignore_subscribe_messages=True)
                pubsub.subscribe(RELOAD_CHANNEL)
                for _ in pubsub.listen():
                    swarm.request_reload()
            except Exception as e:  # Redis restarts must not end reloads for good
                log.warning("specialist reload listener: %s; retrying", e)
                time.sleep(5)

    threading.Thread(target=run, name="dragonfly-swarm-reload", daemon=True).start()


def build_app():
    logging.basicConfig(level=os.environ.get("DRAGONFLY_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    device = os.environ.get("DRAGONFLY_DEVICE") or None
    specialists = os.environ.get("DRAGONFLY_SPECIALISTS")
    engine = load_engine(os.environ.get("DRAGONFLY_CHECKPOINT", "base:answerdotai/ModernBERT-base"), device)
    bank = None
    if os.environ.get("DRAGONFLY_CHECKPOINT_M"):
        # tier M adapters share the M backbone, which then stays unmerged so adapters can be switched in place
        adapters = specialists is not None and has_m_specialists(Path(specialists))
        large = load_engine(os.environ["DRAGONFLY_CHECKPOINT_M"], device, merge=not adapters)
        bank = AdapterBank(large) if adapters else None
        engine = Cascade(engine, large, float(os.environ.get("DRAGONFLY_CASCADE_THRESHOLD", "0.8")))
        log.info("cascade S -> M enabled at confidence < %.2f", engine.threshold)
    if specialists:
        engine = Swarm(engine, specialists, max_loaded=int(os.environ.get("DRAGONFLY_SWARM_MAX_LOADED", "4")),
                       route_threshold=float(os.environ.get("DRAGONFLY_ROUTE_THRESHOLD", "0.5")),
                       load_engine=lambda path: load_engine(path, device), bank=bank)
        if os.environ.get("REDIS_URL"):
            listen_for_reloads(os.environ["REDIS_URL"], engine)
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
    media = MediaResolver(os.environ["PERCEPTION_URL"]) if os.environ.get("PERCEPTION_URL") else None
    return create_app(worker, PluginHost.from_env(), keys=keys, cache=cache, media=media)


def main() -> None:
    uvicorn.run(build_app(), host=os.environ.get("DRAGONFLY_HOST", "0.0.0.0"),
                port=int(os.environ.get("DRAGONFLY_PORT", "8000")), log_level="info", access_log=False)


if __name__ == "__main__":
    main()
