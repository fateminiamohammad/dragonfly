"""Anytime answers over SSE: tier S first, then tier M's upgrades for unsure questions, then the final body."""

import copy
import json

from fastapi.testclient import TestClient

from dragonfly.api import create_app
from dragonfly.batching import Worker
from dragonfly.engine import Cascade

BODY = {"state": "the customer asks about card delivery",
        "questions": {"positive": {"type": "noul", "instructions": "is this positive ?"},
                      "intent": {"type": "choice", "instructions": "which intent",
                                 "criteria": {"card": None, "refund": None, "late": None}}}}


def events(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_stream_sends_fast_answers_then_upgrades(engine):
    cascade = Cascade(engine, copy.deepcopy(engine), {"noul": 0.0, "choice": 1.01})  # only "intent" escalates
    with TestClient(create_app(Worker(cascade))) as c:
        r = c.post("/v1/systemone?stream=true", json=BODY)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        ev = events(r.text)
        plain = c.post("/v1/systemone", json=BODY, headers={"cache-control": "no-cache"}).json()
    assert [e for e, _ in ev] == ["answers", "update", "done"]
    first, update, done = (d for _, d in ev)
    assert set(first["answers"]) == {"positive", "intent"} and not first["final"]
    assert set(update["answers"]) == {"intent"} and update["answers"]["intent"]["tier"] == "M"
    assert done["final"] and done["answers"]["positive"]["tier"] == "S"
    # the final answers equal the ordinary (non-streaming) response
    assert done["answers"]["intent"]["probabilities"] == plain["answers"]["intent"]["probabilities"]
    assert done["answers"]["positive"]["noul"] == plain["answers"]["positive"]["noul"]


def test_stream_without_a_cascade_sends_one_final_event(engine):
    with TestClient(create_app(Worker(engine))) as c:
        ev = events(c.post("/v1/systemone?stream=true", json=BODY).text)
    assert [e for e, _ in ev] == ["done"] and set(ev[0][1]["answers"]) == {"positive", "intent"}


def test_force_large_skips_tier_s(engine):
    large = copy.deepcopy(engine)
    large.config.tier = "M"
    cascade = Cascade(engine, large, 0.0)
    rec = {"state": "a good review", "questions": [{"instr": "positive ?", "options": ["no", "yes"], "qtype": "noul"}]}
    _, _, tiers = cascade.probs([{**rec, "force_large": True}, rec])
    assert tiers == [["M"], ["S"]]
