"""Shared answer cache (DRAGONFLY_CACHE=redis): replicas share answers; a model update changes the namespace."""

import asyncio
import json

import fakeredis

from dragonfly.cache import RedisAnswerCache, model_namespace


def test_replicas_share_answers_and_expire_them():
    redis = fakeredis.FakeAsyncRedis()
    a, b = RedisAnswerCache(redis, "ns1", ttl=60), RedisAnswerCache(redis, "ns1", ttl=60)
    value = ([[0.2, 0.8]], {"tokens": 5, "tiers": [["S"]], "latency_ms": 3.0})

    async def run():
        await a.aput("k", value)
        got = await b.aget("k")  # the other replica: from Redis, then from its local LRU
        again = await b.aget("k")
        ttl = await redis.ttl("dragonfly:answer:ns1:k")
        other = await RedisAnswerCache(redis, "ns2").aget("k")
        return got, again, ttl, other
    got, again, ttl, other = asyncio.run(run())
    assert json.loads(json.dumps(got)) == json.loads(json.dumps(value)) and again == got
    assert b.shared_hits == 1 and 0 < ttl <= 60
    assert other is None  # another model version never sees these answers


def test_redis_outage_is_a_miss_not_an_error():
    class Down:
        async def get(self, key):
            raise ConnectionError("down")

        async def set(self, *a, **k):
            raise ConnectionError("down")
    cache = RedisAnswerCache(Down(), "ns")

    async def run():
        await cache.aput("k", ([[1.0]], {}))
        return await RedisAnswerCache(Down(), "ns").aget("k"), await cache.aget("k")
    assert asyncio.run(run()) == (None, ([[1.0]], {}))


def test_namespace_follows_the_checkpoint_config(tmp_path):
    (tmp_path / "config.json").write_text('{"temperature": 1.0}')
    first = model_namespace([str(tmp_path), ""])
    (tmp_path / "config.json").write_text('{"temperature": 1.2}')
    assert model_namespace([str(tmp_path)]) != first
