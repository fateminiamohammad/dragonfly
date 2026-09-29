"""The Python SDK (sdk/python) against the real API, in-process."""

import asyncio
import copy
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from dragonfly.api import create_app
from dragonfly.batching import Worker
from dragonfly.engine import Cascade

dragonfly_client = pytest.importorskip("dragonfly_client")

STATE = "the customer asks about card delivery"


@pytest.fixture
def client(engine):
    with TestClient(create_app(Worker(Cascade(engine, copy.deepcopy(engine), 0.99)))) as c:
        yield c


def test_decide_batch_stream_and_decorator(client):
    df = dragonfly_client.Dragonfly("http://testserver", transport=client._transport)
    d = df.decide(STATE, positive=bool, intent=Literal["card", "refund"],
                  rating=dragonfly_client.Scale(["one", "two", "three"]),
                  instructions={"positive": "is this positive ?"})
    assert isinstance(bool(d.positive), bool) and d.intent.value in ("card", "refund")
    assert 0 <= d.rating.value <= 2 and d.latency_ms is not None

    results = df.batch([STATE, "a good review"], positive=bool)
    assert len(results) == 2 and all(isinstance(bool(r.positive), bool) for r in results)

    events = list(df.stream(STATE, positive=bool, intent=["card", "refund"]))
    assert events[0][0] == "answers" and events[-1][0] == "done"
    assert events[-1][1].intent.value == d.intent.value

    @df.decision
    def is_positive(review: str) -> bool:
        "is this positive ?"
    assert isinstance(bool(is_positive("a good review")), bool)

    with pytest.raises(dragonfly_client.DragonflyError) as e:
        df.decide(STATE, intent=["card", "refund"], max_error=0.05)  # no risk table on this server
    assert e.value.status == 400


def test_async_client(client):
    import httpx

    class Bridge(httpx.AsyncBaseTransport):
        """Runs the sync test transport from the async client."""

        async def handle_async_request(self, request):
            request.read()
            response = client._transport.handle_request(request)
            response.read()
            return httpx.Response(response.status_code, headers=response.headers, content=response.content)

    async def run():
        async with dragonfly_client.AsyncDragonfly("http://testserver", transport=Bridge()) as df:
            d = await df.decide(STATE, positive=bool)
            events = [e async for e in df.stream(STATE, positive=bool)]
            return d, events
    d, events = asyncio.run(run())
    assert isinstance(bool(d.positive), bool) and events[-1][0] == "done"
