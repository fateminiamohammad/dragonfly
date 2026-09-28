"""Answer cache. Decisions are deterministic (same state, same questions -> same probabilities), so a repeated request
is answered from memory without touching the GPU. Keyed by a hash of the record the model would see, i.e. after
plugins have rewritten the request."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any


def record_key(record: dict, model: str) -> str:
    blob = json.dumps({"m": model, "s": record["state"], "q": [(q["instr"], q["options"]) for q in record["questions"]]},
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class AnswerCache:
    def __init__(self, size: int):
        self.size = size
        self.entries: OrderedDict[str, Any] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        if not self.size:
            return None
        with self.lock:
            value = self.entries.get(key)
            if value is None:
                self.misses += 1
                return None
            self.entries.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value) -> None:
        if not self.size:
            return
        with self.lock:
            self.entries[key] = value
            self.entries.move_to_end(key)
            while len(self.entries) > self.size:
                self.entries.popitem(last=False)

    def stats(self) -> dict:
        return {"size": self.size, "entries": len(self.entries), "hits": self.hits, "misses": self.misses}

    # async interface (the API awaits these, so a shared cache can do I/O without blocking the event loop)
    async def aget(self, key: str):
        return self.get(key)

    async def aput(self, key: str, value) -> None:
        self.put(key, value)


class RedisAnswerCache(AnswerCache):
    """Shared by every model-service replica (DRAGONFLY_CACHE=redis): a local LRU in front of Redis.

    Keys carry a namespace derived from the served checkpoints, so after a model update no replica serves answers
    from the previous model; entries expire after `ttl` seconds. A Redis outage only turns hits into misses."""

    PREFIX = "dragonfly:answer:"

    def __init__(self, redis, namespace: str, size: int = 4096, ttl: int = 86400):
        super().__init__(size)
        self.redis = redis
        self.namespace = namespace
        self.ttl = ttl
        self.shared_hits = 0

    def _key(self, key: str) -> str:
        return f"{self.PREFIX}{self.namespace}:{key}"

    async def aget(self, key: str):
        value = self.get(key)
        if value is not None:
            return value
        try:
            raw = await self.redis.get(self._key(key))
        except Exception:
            return None
        if raw is None:
            return None
        value = tuple(json.loads(raw))
        self.shared_hits += 1
        super().put(key, value)
        return value

    async def aput(self, key: str, value) -> None:
        self.put(key, value)
        try:
            await self.redis.set(self._key(key), json.dumps(value, separators=(",", ":")), ex=self.ttl)
        except Exception:
            pass

    def stats(self) -> dict:
        return {**super().stats(), "shared": True, "shared_hits": self.shared_hits, "namespace": self.namespace}


def model_namespace(paths: list[str]) -> str:
    """Identifies the served models: checkpoint paths plus their config.json contents (temperature included)."""
    from pathlib import Path

    parts = []
    for p in sorted(x for x in paths if x):
        config = Path(p) / "config.json"
        parts.append(p + (config.read_text(encoding="utf-8") if config.is_file() else ""))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]
