"""The swarm: adapter switching, specialist routing, "auto" routing, hot reload, API."""

import copy
import json

import pytest
import torch
from fastapi.testclient import TestClient

from dragonfly.api import create_app
from dragonfly.batching import Worker
from dragonfly.swarm import AdapterBank, Swarm

STATE = "the customer asks about card delivery . the payment was late"
Q = {"instr": "which intent best describes this message", "options": ["card arrived", "refund", "payment late"],
     "qtype": "choice"}


def rec(**extra):
    return {"state": STATE, "questions": [dict(Q)], **extra}


def card(folder, name, tier="S", description=None):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"tier": tier, "backbone": "tiny"}))
    (folder / "specialist.json").write_text(json.dumps({"name": name, "description": description or name}))


def perturbed(engine, seed):
    """A copy of a tier M engine with different adapters and head: what training another specialist would give."""
    other = copy.deepcopy(engine)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in other.model.named_parameters():
            if "lora_" in name or name.startswith("head."):
                p.add_(torch.randn(p.shape, generator=g) * 0.05)
    other.config.temperature = 1.7
    return other


def test_adapter_switch_equals_loading_it(decoder_engine, tmp_path):
    general = decoder_engine.probs([rec()])[0]
    other = perturbed(decoder_engine, 1)
    expected = other.probs([rec()])[0]
    other.save(str(tmp_path / "invoices"))

    bank = AdapterBank(decoder_engine)
    bank.add("invoices", str(tmp_path / "invoices"))
    bank.activate("invoices")
    assert decoder_engine.probs([rec()])[0][0][0] == pytest.approx(expected[0][0], abs=1e-5)
    assert decoder_engine.config.temperature == 1.7
    bank.activate("general")
    assert decoder_engine.probs([rec()])[0][0][0] == pytest.approx(general[0][0], abs=1e-5)
    assert bank.switches == 2


def test_adapter_for_another_base_is_refused(decoder_engine, tmp_path):
    other = perturbed(decoder_engine, 2)
    other.config.lora_r = 8
    other.save(str(tmp_path / "x"))
    with pytest.raises(ValueError, match="r=8"):
        AdapterBank(decoder_engine).add("x", str(tmp_path / "x"))


def test_swarm_routes_by_model_and_mixes_in_one_batch(engine, tmp_path):
    card(tmp_path / "invoices", "invoices")
    special = copy.deepcopy(engine)
    with torch.no_grad():
        special.model.head.q.weight.mul_(-3)
    loads = []
    swarm = Swarm(engine, str(tmp_path), load_engine=lambda path: loads.append(path) or special)
    records = [rec(), rec(specialist="invoices"), rec(specialist="dragonfly-latest")]
    probs, _, tiers = swarm.probs(records)
    assert probs[0] == probs[2] == engine.probs([rec()])[0][0]
    assert probs[1] == special.probs([rec()])[0][0]
    assert [r["routed_to"] for r in records] == ["general", "invoices", "general"]
    swarm.probs([rec(specialist="invoices")])
    assert len(loads) == 1  # loaded once, then kept
    with pytest.raises(ValueError, match="unknown model"):
        swarm.probs([rec(specialist="nope")])


def test_least_recently_used_specialists_are_unloaded(engine, tmp_path):
    for n in ("a", "b", "c"):
        card(tmp_path / n, n)
    swarm = Swarm(engine, str(tmp_path), max_loaded=2, load_engine=lambda path: engine)
    for n in ("a", "b", "a", "c"):
        swarm.probs([rec(specialist=n)])
    assert list(swarm.loaded) == ["a", "c"]


class Router:
    """A fake general tier S whose answer to every question is fixed."""

    def __init__(self, p):
        self.p, self.config, self.device = p, None, "cpu"

    def probs(self, records):
        return [[list(self.p) for _ in r["questions"]] for r in records], [1] * len(records), \
            [["S"] * len(r["questions"]) for r in records]

    def describe(self):
        return {"tier": "S", "trained": True}


def test_auto_picks_a_specialist_only_when_the_router_is_sure(tmp_path):
    card(tmp_path / "invoices", "invoices", description="invoices and payments")
    card(tmp_path / "support", "support", description="customer support tickets")
    sure = Swarm(Router([0.02, 0.96, 0.02]), str(tmp_path), load_engine=lambda path: Router([1.0]))
    r = rec(specialist="auto")
    sure.probs([r])
    assert r["routed_to"] == "support"
    unsure = Swarm(Router([0.4, 0.35, 0.25]), str(tmp_path), load_engine=lambda path: Router([1.0]))
    r = rec(specialist="auto")
    unsure.probs([r])
    assert r["routed_to"] == "general"
    none = Swarm(Router([0.01, 0.01, 0.98]), str(tmp_path), load_engine=lambda path: Router([1.0]))
    r = rec(specialist="auto")
    none.probs([r])
    assert r["routed_to"] == "general"  # "none of these"


def test_hot_reload_picks_up_new_and_removed_specialists(engine, tmp_path):
    swarm = Swarm(engine, str(tmp_path), load_engine=lambda path: engine)
    assert swarm.list() == []
    card(tmp_path / "invoices", "invoices")
    swarm.request_reload()
    swarm.probs([rec()])
    assert [s["name"] for s in swarm.list()] == ["invoices"]
    (tmp_path / "invoices" / "specialist.json").unlink()
    swarm.request_reload()
    swarm.probs([rec()])
    assert swarm.list() == []


def test_api_lists_specialists_and_reports_the_route(engine, tmp_path):
    card(tmp_path / "invoices", "invoices", description="invoices and payments")
    swarm = Swarm(engine, str(tmp_path), load_engine=lambda path: engine)
    request = {"state": STATE, "model": "invoices",
               "questions": {"q": {"type": "noul", "instructions": "is this about a payment ?"}}}
    with TestClient(create_app(Worker(swarm))) as c:
        assert c.get("/v1/specialists").json()["specialists"][0]["name"] == "invoices"
        names = [m["name"] for m in c.get("/v1/models").json()["models"]]
        assert "invoices" in names and "auto" in names and "dragonfly-latest" in names
        r = c.post("/v1/systemone", json=request)
        assert r.status_code == 200, r.text
        assert r.json()["specialist"] == "invoices"
        assert c.post("/v1/systemone", json={**request, "model": "dragonfly-latest"}).json()["specialist"] == "general"
        assert c.post("/v1/systemone", json={**request, "model": "nope"}).status_code == 422


def test_listing_shows_new_specialists_before_the_model_thread_reloads(engine, tmp_path):
    swarm = Swarm(engine, str(tmp_path), load_engine=lambda path: engine)
    card(tmp_path / "invoices", "invoices")
    swarm.request_reload()
    assert [s["name"] for s in swarm.list()] == ["invoices"]  # read from disk, nothing loaded yet
    assert swarm.cards == {}


def test_specialists_preload_in_the_background_and_retraining_replaces_them(engine, tmp_path):
    import time

    card(tmp_path / "invoices", "invoices")
    loads, prepared = [], []

    def load(path):
        loads.append(path)
        return copy.deepcopy(engine)

    swarm = Swarm(engine, str(tmp_path), load_engine=load, prepare=lambda e: prepared.append(e) or e)
    deadline = time.time() + 5
    while not swarm._ready and time.time() < deadline:
        time.sleep(0.01)
    assert "invoices" in swarm._ready and len(loads) == 1  # read on the preload thread
    swarm.probs([rec(specialist="invoices")])
    assert len(loads) == 1 and len(prepared) == 1  # the first request used it: no load on the model thread

    spec = json.loads((tmp_path / "invoices" / "specialist.json").read_text())
    (tmp_path / "invoices" / "specialist.json").write_text(json.dumps({**spec, "job": "retrained"}))
    swarm.request_reload()
    swarm.probs([rec()])
    assert "invoices" not in swarm.loaded  # the old version is dropped
    assert swarm.version("invoices") == "@retrained"
