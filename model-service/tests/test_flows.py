"""Agent flows: conditions, state templates, validation, execution, and the /v1/flows API."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from dragonfly.api import create_app
from dragonfly.batching import Worker
from dragonfly.flows import FlowError, mermaid, parse_condition, render_state, run_flow, validate

FLOW = {
    "name": "triage",
    "start": "classify",
    "steps": {
        "classify": {
            "questions": {"intent": {"type": "choice", "criteria": {"refund": None, "delivery": None}}},
            "next": [{"if": "intent.choice == refund and intent.confidence > 0.5", "to": "risk"},
                     {"to": "delivery"}],
        },
        "risk": {"model": "fraud", "state": "{{state}} | intent={{classify.intent.choice}}",
                 "questions": {"risky": {"type": "noul", "instructions": "risky?"}}},
        "delivery": {"questions": {"late": {"type": "noul", "instructions": "late?"}}},
    },
}


def test_conditions_and_bind_tighter_than_or():
    cond = parse_condition("a.choice == x and a.confidence > 0.5 or b.noul >= 0.9")
    assert cond({"a": {"choice": "x", "confidence": 0.6}})
    assert not cond({"a": {"choice": "x", "confidence": 0.4}})
    assert cond({"a": {"choice": "y", "confidence": 0.1}, "b": {"noul": 0.95}})
    assert parse_condition("b.value == true")({"b": {"noul": 0.8}})
    assert not parse_condition("b.value == true")({"b": {"noul": 0.2}})
    assert parse_condition(None)({})
    with pytest.raises(FlowError):
        parse_condition("nonsense")


def test_state_templates():
    history = {"classify": {"intent": {"choice": "refund"}}}
    assert render_state(None, "doc", history) == "doc"
    assert render_state("{{state}} -> {{classify.intent.choice}}", "doc", history) == "doc -> refund"
    assert render_state("{{ classify.intent.missing }}!", "doc", history) == "!"
    with pytest.raises(FlowError):
        render_state("{{classify.intent}}", "doc", history)


def test_validation_catches_bad_edges_up_front():
    with pytest.raises(FlowError, match="undefined step"):
        validate({"steps": {"a": {"questions": {"q": {}}, "next": [{"to": "b"}]}}})
    with pytest.raises(FlowError, match="needs questions"):
        validate({"steps": {"a": {}}})
    assert validate({"steps": {"a": {"questions": {"q": {}}}}})["start"] == "a"


def fake_decide(answers_by_question):
    seen = []

    async def decide(body):
        seen.append(body)
        return {"answers": {q: answers_by_question[q] for q in body["questions"]}, "latency_ms": 1.0,
                "specialist": body.get("model", "general")}
    return decide, seen


def test_run_follows_the_first_matching_edge_and_maps_state():
    decide, seen = fake_decide({"intent": {"choice": "refund", "confidence": 0.9}, "risky": {"noul": 0.2}})
    out = asyncio.run(run_flow(FLOW, "charged twice", decide))
    assert out["path"] == ["classify", "risk"]
    assert seen[1] == {"state": "charged twice | intent=refund", "questions": FLOW["steps"]["risk"]["questions"],
                       "model": "fraud"}
    assert out["steps"][1]["specialist"] == "fraud" and out["steps"][-1]["next"] is None

    decide, _ = fake_decide({"intent": {"choice": "refund", "confidence": 0.3}, "late": {"noul": 0.7}})
    assert asyncio.run(run_flow(FLOW, "x", decide))["path"] == ["classify", "delivery"]  # the fallback edge


def test_loops_are_bounded():
    loop = {"steps": {"a": {"questions": {"q": {}}, "next": [{"to": "a"}]}}}
    decide, _ = fake_decide({"q": {"noul": 0.5}})
    with pytest.raises(FlowError, match="within 3 steps"):
        asyncio.run(run_flow(loop, "x", decide, max_steps=3))


def test_mermaid_highlights_the_path():
    text = mermaid(FLOW, ["classify", "risk"])
    assert text.startswith("flowchart LR") and "classify -- " in text and "class classify,risk taken" in text


class Store:
    def __init__(self, flows):
        self.flows = flows

    async def get(self, name):
        return self.flows.get(name)


def test_flow_api_runs_saved_and_inline_flows(engine):
    flow = {"steps": {
        "first": {"questions": {"positive": {"type": "noul", "instructions": "is this positive ?"}},
                  "next": [{"if": "positive.noul >= 0", "to": "second"}]},
        "second": {"state": "{{state}} again", "questions": {"rating": {"type": "score", "criteria": ["bad", "good"]}}},
    }}
    with TestClient(create_app(Worker(engine), flows=Store({"demo": flow}))) as c:
        r = c.post("/v1/flows/demo/run", json={"state": "a good review", "mermaid": True})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["path"] == ["first", "second"] and "rating" in body["answers"]["second"]
        assert "class first,second taken" in body["mermaid"]
        assert c.post("/v1/flows/run", json={"flow": flow, "state": "x"}).json()["path"] == ["first", "second"]
        assert c.post("/v1/flows/nope/run", json={"state": "x"}).status_code == 404
        bad = {"steps": {"a": {"questions": {"q": {"type": "noul"}}, "next": [{"to": "zzz"}]}}}
        assert c.post("/v1/flows/run", json={"flow": bad, "state": "x"}).status_code == 422
