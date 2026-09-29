"""Batch / map-reduce: many decisions in one call, results in order, bad rows isolated, live traffic first."""

import threading
import time

from fastapi.testclient import TestClient

from dragonfly.api import create_app
from dragonfly.batching import Worker


def req(i, state=None):
    return {"state": state or f"the customer asks about card delivery {'. the payment was late' * (i % 4)}",
            "questions": {"positive": {"type": "noul", "instructions": "is this positive ?"},
                          "rating": {"type": "score", "instructions": "rating", "criteria": ["one", "two", "three"]}}}


def test_batch_answers_every_request_in_order(engine):
    requests = [req(i) for i in range(40)] + [{"state": "x", "questions": {}}]  # the last one is invalid
    with TestClient(create_app(Worker(engine))) as c:
        r = c.post("/v1/batch", json={"requests": requests})
        assert r.status_code == 200, r.text
        body = r.json()
        single = [c.post("/v1/systemone", json=q).json()["answers"] for q in requests[:3]]
    results = body["results"]
    assert len(results) == 41 and "error" in results[-1]
    for got, want in zip(results[:3], single):  # same answers as asking one by one, in the same order
        assert got["answers"]["positive"]["noul"] == want["positive"]["noul"]
    assert body["stats"]["requests"] == 41 and body["stats"]["errors"] == 1 and body["stats"]["questions"] == 80


def test_batch_jobs_run_in_the_background(engine):
    with TestClient(create_app(Worker(engine))) as c:
        job = c.post("/v1/batch/jobs", json={"requests": [req(i) for i in range(20)]}).json()
        deadline = time.time() + 20
        while True:
            state = c.get(f"/v1/batch/jobs/{job['id']}").json()
            if state["status"] != "running" or time.time() > deadline:
                break
            time.sleep(0.05)
        assert state["status"] == "done" and state["done"] == 20 and len(state["results"]) == 20
        assert "results" not in c.get(f"/v1/batch/jobs/{job['id']}?results=false").json()
        assert c.get("/v1/batch/jobs/nope").status_code == 404
        assert c.post("/v1/batch", json={"requests": []}).status_code == 422


class SlowEngine:
    """Records what each forward pass contained; the first pass blocks until released."""

    def __init__(self):
        self.batches = []
        self.release = threading.Event()

    def probs(self, records):
        if not self.batches:
            self.release.wait(5)
        self.batches.append([r["tag"] for r in records])
        return [[[1.0]] for _ in records], [1] * len(records), [["S"] for _ in records]


def test_live_requests_go_before_queued_batch_work():
    engine = SlowEngine()
    worker = Worker(engine, max_batch=4)
    first = worker.submit({"tag": "first"})
    time.sleep(0.1)  # the worker is now busy with "first"
    low = [worker.submit({"tag": f"batch{i}"}, low_priority=True) for i in range(10)]
    live = worker.submit({"tag": "live"})
    engine.release.set()
    for f in [first, live, *low]:
        f.result(5)
    worker.close()
    assert engine.batches[1][0] == "live"  # the next pass starts with the live request
    assert sum(len(b) for b in engine.batches) == 12
