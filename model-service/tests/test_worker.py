"""Trainer worker: job lifecycle (queue -> warm-start training -> held-out metrics -> registered specialist)."""

import json

import fakeredis
import pytest

from dragonfly.swarm import read_card
from dragonfly.worker import JOB, JOBS, TrainerWorker, split


def labelled(n):
    rows = []
    for i in range(n):
        good = i % 2 == 0
        rows.append(json.dumps({"state": "a good review" if good else "the payment was late",
                                "questions": {"q": {"type": "noul", "instructions": "is the customer positive ?",
                                                    "label": good}}}))
    return "\n".join(rows) + "\n"


@pytest.fixture
def setup(engine, tmp_path):
    engine.save(str(tmp_path / "general-s"))
    (tmp_path / "uploads").mkdir()
    redis = fakeredis.FakeRedis()
    worker = TrainerWorker(redis, str(tmp_path / "specialists"), str(tmp_path / "uploads"),
                           {"S": str(tmp_path / "general-s"), "M": None}, max_length=64, device="cpu")
    return worker, redis, tmp_path


def job_state(redis, job_id):
    return {k.decode(): json.loads(v) for k, v in redis.hgetall(JOB.format(job_id)).items()}


def test_split_is_deterministic_and_disjoint():
    lines = [str(i) for i in range(100)]
    train, cal, test = split(lines)
    assert (len(train), len(cal), len(test)) == (70, 10, 20)
    assert set(train) | set(cal) | set(test) == set(lines)
    assert split(lines) == (train, cal, test)


def test_upload_becomes_a_registered_specialist(setup):
    worker, redis, tmp = setup
    (tmp / "uploads" / "j1.jsonl").write_text(labelled(60))
    pubsub = redis.pubsub()
    pubsub.subscribe("dragonfly:specialists")
    redis.lpush(JOBS, json.dumps({"id": "j1", "op": "train", "name": "reviews", "description": "product reviews",
                                  "tier": "S", "data": str(tmp / "uploads" / "j1.jsonl"), "epochs": 1}))
    assert worker.run_once(timeout=1)
    state = job_state(redis, "j1")
    assert state["status"] == "done", state
    assert state["metrics"]["test_questions"] == 12 and 0 <= state["metrics"]["accuracy"] <= 1
    card = read_card(tmp / "specialists" / "reviews")
    assert card["description"] == "product reviews" and card["tier"] == "S"
    assert card["metrics"]["train_questions"] == 42
    assert not list((tmp / "specialists").glob(".*"))  # staging folder renamed, nothing left behind
    messages = [m for m in iter(pubsub.get_message, None) if m["type"] == "message"]
    assert json.loads(messages[0]["data"]) == {"added": "reviews"}

    redis.lpush(JOBS, json.dumps({"id": "j2", "op": "delete", "name": "reviews"}))
    worker.run_once(timeout=1)
    assert not (tmp / "specialists" / "reviews").exists()


@pytest.mark.parametrize("job, error", [
    ({"name": "auto"}, "invalid specialist name"),
    ({"name": "Bad Name"}, "invalid specialist name"),
    ({"name": "ok-name", "tier": "M"}, "tier M training is disabled"),
    ({"name": "ok-name", "rows": 10}, "at least 50"),
    ({"name": "ok-name", "data": "/etc/passwd"}, "inside the uploads folder"),
])
def test_bad_jobs_fail_with_a_reason(setup, job, error):
    worker, redis, tmp = setup
    (tmp / "uploads" / "x.jsonl").write_text(labelled(job.pop("rows", 60)))
    redis.lpush(JOBS, json.dumps({"id": "x", "op": "train", "tier": "S", "data": str(tmp / "uploads" / "x.jsonl"),
                                  **job}))
    worker.run_once(timeout=1)
    state = job_state(redis, "x")
    assert state["status"] == "failed" and error in state["error"]
