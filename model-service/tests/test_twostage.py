"""Two-stage choice (> 255 options): the anchor puts chunks on one scale; the API prunes then decides."""

import math
import random

import pytest
from fastapi.testclient import TestClient

from dragonfly.api import create_app
from dragonfly.batching import Worker
from dragonfly.twostage import chunks, finalists, global_scores


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [v / sum(e) for v in e]


def test_anchor_makes_chunk_scores_equal_to_one_global_pass():
    rng = random.Random(0)
    keys = [f"k{i}" for i in range(700)]
    logit = {k: rng.gauss(0, 3) for k in keys}  # an isolated-option scorer: a key's logit ignores its neighbours
    parts = chunks(keys)
    assert all(p[0] == "k0" for p in parts) and all(len(p) <= 255 for p in parts)
    assert sorted({k for p in parts for k in p}) == sorted(keys)
    scores = global_scores(parts, [softmax([logit[k] for k in p]) for p in parts])
    for k in keys:
        assert scores[k] == pytest.approx(logit[k] - logit["k0"], abs=1e-6)
    top, pruned = finalists(scores, 32)
    assert top == sorted(keys, key=logit.get, reverse=True)[:32]
    full = softmax([logit[k] for k in keys])
    kept = sum(p for k, p in zip(keys, full) if k in top)
    assert pruned == pytest.approx(1 - kept, abs=1e-6)


def test_api_answers_a_600_option_question(engine):
    criteria = {f"team{i}": None for i in range(600)}
    body = {"state": "the customer asks about card delivery",
            "questions": {"route": {"type": "choice", "instructions": "which team", "criteria": criteria},
                          "positive": {"type": "noul", "instructions": "is this positive ?"}}}
    with TestClient(create_app(Worker(engine))) as c:
        r = c.post("/v1/systemone", json=body)
        assert r.status_code == 200, r.text
        a = r.json()["answers"]["route"]
        assert a["two_stage"]["options"] == 600 and a["two_stage"]["candidates"] == 32
        assert len(a["probabilities"]) == 32 and a["choice"] in criteria
        assert 0 <= a["two_stage"]["pruned_mass"] <= 1
        assert "two_stage" not in r.json()["answers"]["positive"]
        too_many = {**body, "questions": {"route": {"type": "choice", "criteria": {f"t{i}": None for i in range(10001)}}}}
        assert c.post("/v1/systemone", json=too_many).status_code == 422
