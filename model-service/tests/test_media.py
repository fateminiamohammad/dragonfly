import httpx
import pytest
from fastapi.testclient import TestClient

from dragonfly.api import create_app
from dragonfly.batching import Worker
from dragonfly.cache import AnswerCache
from dragonfly.media import MediaResolver, find_media, text_form
from dragonfly.plugins import Plugin, PluginHost

QUESTIONS = {"q": {"type": "choice", "criteria": {"card": None, "refund": None}}}


def fake_perception(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        body = __import__("json").loads(request.content)
        seen.append(body)
        items = {}
        for it in body["items"]:
            if it["type"] == "image":
                tags = [{"label": "invoice", "score": 0.91, "p": 0.5}, {"label": "document", "score": 0.08, "p": 0.4}]
                items[it["id"]] = {"type": "image", "ocr": {"text": "INVOICE 42", "lines": []}, "tags": tags}
            elif it["data"] == "bad":
                return httpx.Response(422, json={"detail": "item '0': could not decode audio"})
            else:
                items[it["id"]] = {"type": "audio", "transcript": "my card has not arrived", "language": it.get("language", "en")}
        return httpx.Response(200, json={"items": items, "latency_ms": {"total": 12.5}})
    return handler


@pytest.fixture
def seen():
    return []


@pytest.fixture
def resolver(seen):
    return MediaResolver("http://perception", client=httpx.AsyncClient(transport=httpx.MockTransport(fake_perception(seen))))


def test_find_media_walks_nested_state():
    state = {"msg": "hi", "files": [{"type": "image", "data": "x"}, {"type": "note"}], "voice": {"type": "audio", "data": "y"}}
    assert [p for p, _ in find_media(state)] == [("files", 0), ("voice",)]
    assert find_media({"type": "image", "data": "x"}) == [((), {"type": "image", "data": "x"})]
    assert find_media({"type": "image", "url": "http://x"}) == []  # URLs are never fetched


def test_text_form_keeps_names_and_drops_data():
    img = text_form({"type": "image", "data": "x", "name": "scan.png", "tasks": ["ocr"]},
                    {"ocr": {"text": "A"}, "tags": [{"label": "car", "score": 0.9, "p": 0.2}]})
    assert img == {"type": "image", "name": "scan.png", "text": "A", "shows": "car (0.90)"}


def test_decision_on_image_and_voice(engine, resolver, seen):
    class Capture(Plugin):
        name = "capture"

        def on_request(self, request):
            self.state = request.state  # plugins see text, never raw media
            return request

    cap = Capture()
    with TestClient(create_app(Worker(engine), plugins=PluginHost([cap]), media=resolver)) as c:
        r = c.post("/v1/systemone", json={"state": {"photo": {"type": "image", "data": "aW1n", "name": "a.png"},
                                                    "voice": {"type": "audio", "data": "YXVk", "language": "fa"}},
                                          "questions": QUESTIONS})
    assert r.status_code == 200, r.text
    assert r.json()["usage"]["media_ms"] == 12.5
    assert len(seen) == 1 and [i["type"] for i in seen[0]["items"]] == ["image", "audio"]  # one batched call
    assert seen[0]["items"][1]["language"] == "fa"
    assert cap.state == {"photo": {"type": "image", "name": "a.png", "text": "INVOICE 42",
                                   "shows": "invoice (0.91), document (0.08)"},
                         "voice": {"type": "audio", "transcript": "my card has not arrived"}}


def test_repeated_media_hits_the_answer_cache(engine, resolver):
    worker = Worker(engine)
    req = {"state": {"type": "image", "data": "aW1n"}, "questions": QUESTIONS}
    with TestClient(create_app(worker, media=resolver, cache=AnswerCache(10))) as c:
        assert not c.post("/v1/systemone", json=req).json()["cached"]
        assert c.post("/v1/systemone", json=req).json()["cached"]
    assert worker.requests == 1


def test_media_without_perception_service_is_422(engine):
    with TestClient(create_app(Worker(engine))) as c:
        r = c.post("/v1/systemone", json={"state": {"type": "audio", "data": "x"}, "questions": QUESTIONS})
        assert r.status_code == 422 and "PERCEPTION_URL" in r.text
        assert c.post("/v1/perceive", json={"items": []}).status_code == 404


def test_perception_errors_pass_through(engine, resolver):
    with TestClient(create_app(Worker(engine), media=resolver)) as c:
        r = c.post("/v1/systemone", json={"state": {"type": "audio", "data": "bad"}, "questions": QUESTIONS})
    assert r.status_code == 422 and "decode audio" in r.text


def test_perceive_proxy_requires_key(engine, resolver):
    with TestClient(create_app(Worker(engine), api_keys=["k"], media=resolver)) as c:
        body = {"items": [{"id": "1", "type": "image", "data": "aW1n"}]}
        assert c.post("/v1/perceive", json=body).status_code == 401
        ok = c.post("/v1/perceive", json=body, headers={"authorization": "Bearer k"})
        assert ok.status_code == 200 and ok.json()["items"]["1"]["ocr"]["text"] == "INVOICE 42"


def test_unreachable_perception_is_503(engine):
    def down(request):
        raise httpx.ConnectError("refused")
    resolver = MediaResolver("http://perception", client=httpx.AsyncClient(transport=httpx.MockTransport(down)))
    with TestClient(create_app(Worker(engine), media=resolver)) as c:
        r = c.post("/v1/systemone", json={"state": {"type": "image", "data": "x"}, "questions": QUESTIONS})
    assert r.status_code == 503
