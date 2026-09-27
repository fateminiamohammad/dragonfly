"""API keys and usage, shared with the backend through Redis (both optional).

Keys: the backend stores SHA-256 hex digests of active keys in the Redis set `dragonfly:keys`. The model-service
checks a presented key's digest against static keys from DRAGONFLY_API_KEYS and then Redis. Results are cached
in-process for `ttl` seconds, so a hot key costs no network round trip. Revoking a key takes effect within `ttl`.

Usage: after each answered request the service increments `dragonfly:usage:<digest>:<YYYY-MM-DD>` (a hash with
requests / questions / tokens). The backend reads these for its usage pages. Writing happens after the response is
computed and never fails a request.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from datetime import UTC, datetime

log = logging.getLogger(__name__)
KEYS_SET = "dragonfly:keys"


def digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class KeyStore:
    def __init__(self, static_keys: list[str] | None = None, redis=None, ttl: float = 30.0):
        self.static = [digest(k) for k in (static_keys or [])]
        self.redis = redis  # a redis.asyncio.Redis, or None
        self.ttl = ttl
        self.cache: dict[str, tuple[bool, float]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.static) or self.redis is not None

    async def check(self, key: str) -> str | None:
        """-> the key's digest when valid, else None."""
        if not key:
            return None
        d = digest(key)
        if any(hmac.compare_digest(d, s) for s in self.static):
            return d
        if self.redis is None:
            return None
        hit = self.cache.get(d)
        now = time.monotonic()
        if hit and hit[1] > now:
            return d if hit[0] else None
        try:
            ok = bool(await self.redis.sismember(KEYS_SET, d))
        except Exception:
            log.exception("redis unavailable for key check")
            return d if hit and hit[0] else None  # keep serving keys we saw recently; reject unknown ones
        self.cache[d] = (ok, now + self.ttl)
        return d if ok else None

    async def record_usage(self, key_digest: str, questions: int, tokens: int) -> None:
        if self.redis is None or not key_digest:
            return
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        name = f"dragonfly:usage:{key_digest}:{day}"
        try:
            pipe = self.redis.pipeline()
            pipe.hincrby(name, "requests", 1)
            pipe.hincrby(name, "questions", questions)
            pipe.hincrby(name, "tokens", tokens)
            pipe.expire(name, 90 * 86400)
            await pipe.execute()
        except Exception:
            log.warning("usage not recorded (redis unavailable)")
