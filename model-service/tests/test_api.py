import pytest
from fastapi.testclient import TestClient

from dragonfly.api import create_app
from dragonfly.batching import Worker
from dragonfly.plugins import Plugin, PluginError, PluginHost

REQUEST = {
    "state": "the customer asks about card delivery",
    "questions": {
        "intent": {"type": "choice", "instructions": "which intent", "criteria": {"card": None, "refund": None}},
        "positive": {"type": "noul", "instructions": "is this positive ?"},
        "rating": {"type": "score", "instructions": "rating", "criteria": ["one", "two", "three"]},
    },
}


@pytest.fixture
def client_factory(engine):
    def make(**kwargs):
        return TestClient(create_app(Worker(engine), **kwargs))
    return make


def test_systemone_answers_every_question(client_factory):
    with client_factory() as c:
        r = c.post("/v1/systemone", json=REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["answers"]) == {"intent", "positive", "rating"}
    assert body["answers"]["intent"]["choice"] in {"card", "refund"}
    assert 0 <= body["answers"]["positive"]["noul"] <= 1
    assert body["usage"]["output_tokens"] == 0
    assert r.headers["x-typesafe-request-id"]


def test_decide_is_an_alias(client_factory):
    with client_factory() as c:
        assert c.post("/v1/decide", json=REQUEST).status_code == 200


def test_api_key_required_when_configured(client_factory):
    with client_factory(api_keys=["secret"]) as c:
        assert c.post("/v1/systemone", json=REQUEST).status_code == 401
        ok = c.post("/v1/systemone", json=REQUEST, headers={"authorization": "Bearer secret"})
        assert ok.status_code == 200
        assert c.get("/health").status_code == 200  # health stays open for the orchestrator


def test_invalid_request_is_422(client_factory):
    with client_factory() as c:
        assert c.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 422


def test_models_and_metrics(client_factory):
    with client_factory() as c:
        c.post("/v1/systemone", json=REQUEST)
        names = [m["name"] for m in c.get("/v1/models").json()["models"]]
        assert "jev-latest" in names and "dragonfly-latest" in names
        assert "dragonfly_requests_total" in c.get("/metrics").text


class Upper(Plugin):
    name = "upper"

    def on_decision(self, request, response):
        response["plugin"] = "seen"
        return response


class Deny(Plugin):
    name = "deny"

    def on_request(self, request):
        if "refund" in str(request.state):
            raise PluginError("refund requests go to a human", 403)
        return request


def test_plugins_run_on_the_request_path(client_factory):
    with client_factory(plugins=PluginHost([Upper(), Deny()])) as c:
        assert c.post("/v1/systemone", json=REQUEST).json()["plugin"] == "seen"
        blocked = {**REQUEST, "state": "i want a refund"}
        r = c.post("/v1/systemone", json=blocked)
        assert r.status_code == 403 and "human" in r.text
