import asyncio

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from dragonfly.api import create_app
from dragonfly.auth import KEYS_SET, KeyStore, digest
from dragonfly.batching import Worker
from dragonfly.cache import AnswerCache
from dragonfly.engine import Cascade

REQUEST = {"state": "the customer asks about card delivery",
           "questions": {"intent": {"type": "choice", "criteria": {"card": None, "refund": None, "payment": None}},
                         "positive": {"type": "noul", "instructions": "is this good ?"}}}
RECORD = {"state": "the customer asks about card delivery",
          "questions": [{"instr": "which intent", "options": ["card", "refund", "payment"], "qtype": "choice"},
                        {"instr": "is this good ?", "options": ["no", "yes"], "qtype": "noul"}]}


def test_cascade_threshold_zero_never_escalates(engine, decoder_engine):
    probs, _, tiers = Cascade(engine, decoder_engine, threshold=0.0).probs([RECORD])
    assert tiers == [["S", "S"]]
    assert probs == engine.probs([RECORD])[0]


def test_cascade_threshold_one_escalates_everything(engine, decoder_engine):
    c = Cascade(engine, decoder_engine, threshold=1.01)
    probs, _, tiers = c.probs([RECORD])
    assert tiers == [["M", "M"]]
    assert probs[0][1] == pytest.approx(decoder_engine.probs([RECORD])[0][0][1], abs=1e-5)
    assert c.describe()["escalation_rate"] == 1.0


def test_answer_cache_hits_skip_the_model(engine):
    worker = Worker(engine)
    with TestClient(create_app(worker, cache=AnswerCache(10))) as c:
        first = c.post("/v1/systemone", json=REQUEST).json()
        second = c.post("/v1/systemone", json=REQUEST).json()
    assert not first["cached"] and second["cached"]
    assert first["answers"] == second["answers"]
    assert worker.requests == 1


def test_answers_report_their_tier(engine):
    with TestClient(create_app(Worker(engine))) as c:
        answers = c.post("/v1/systemone", json=REQUEST).json()["answers"]
    assert {a["tier"] for a in answers.values()} == {"S"}


def test_cache_evicts_least_recently_used():
    cache = AnswerCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1


def test_redis_keys_and_usage(engine):
    redis = fakeredis.aioredis.FakeRedis()
    asyncio.run(redis.sadd(KEYS_SET, digest("backend-issued")))
    with TestClient(create_app(Worker(engine), keys=KeyStore([], redis))) as c:
        assert c.post("/v1/systemone", json=REQUEST).status_code == 401
        assert c.post("/v1/systemone", json=REQUEST, headers={"authorization": "Bearer nope"}).status_code == 401
        ok = c.post("/v1/systemone", json=REQUEST, headers={"authorization": "Bearer backend-issued"})
        assert ok.status_code == 200
    usage = asyncio.run(redis.keys(f"dragonfly:usage:{digest('backend-issued')}:*"))
    assert len(usage) == 1
    counts = asyncio.run(redis.hgetall(usage[0]))
    assert counts[b"requests"] == b"1" and counts[b"questions"] == b"2"
