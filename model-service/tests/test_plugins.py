import pytest

from dragonfly.plugins import Plugin, PluginHost
from dragonfly.plugins import host as host_module
from dragonfly.plugins.api import compatible
from dragonfly.schema import DecideRequest

REQ = DecideRequest.model_validate({"state": "x", "questions": {"q": {"type": "noul"}}})


class Escalate(Plugin):
    name = "escalate"

    def on_low_confidence(self, request, question_id, answer):
        return {**answer, "noul": 1.0, "confidence": 1.0, "escalated": True}


class Broken(Plugin):
    name = "broken"
    fail_open = True

    def on_decision(self, request, response):
        raise RuntimeError("boom")


def test_version_compatibility():
    assert compatible("1.0", "1.2")
    assert not compatible("1.3", "1.2")
    assert not compatible("2.0", "1.9")


def test_low_confidence_hook_replaces_answer():
    host = PluginHost([Escalate()], low_confidence=0.5)
    resp = {"answers": {"q": {"type": "noul", "noul": 0.55, "confidence": 0.1}}}
    assert host.on_decision(REQ, resp)["answers"]["q"]["escalated"]


def test_fail_open_plugin_is_skipped():
    resp = {"answers": {}}
    assert PluginHost([Broken()]).on_decision(REQ, resp) is resp


def test_fail_closed_plugin_raises():
    class Strict(Broken):
        fail_open = False
    with pytest.raises(RuntimeError):
        PluginHost([Strict()]).on_decision(REQ, {"answers": {}})


def test_discover_uses_entry_points_and_rejects_unknown(monkeypatch):
    class EP:
        name = "escalate"

        def load(self):
            return Escalate
    monkeypatch.setattr(host_module, "entry_points", lambda group: [EP()])
    assert [p.name for p in host_module.discover(["escalate"])] == ["escalate"]
    with pytest.raises(RuntimeError, match="not installed"):
        host_module.discover(["missing"])


def test_plugin_config_from_env():
    env = {"DRAGONFLY_PLUGIN_REDACT_PII_MODE": "strict", "OTHER": "x"}
    assert host_module.plugin_config("redact-pii", env) == {"mode": "strict"}


def test_blocking_plugin_runs_off_the_event_loop_and_times_out():
    import asyncio
    import threading
    import time

    loop_thread = {}

    class Slow(Plugin):
        name = "slow"
        blocking = True
        timeout_s = 0.2
        fail_open = True

        def on_request(self, request):
            loop_thread["hook"] = threading.get_ident()
            time.sleep(1.0)  # longer than the budget
            return request

    async def run():
        loop_thread["loop"] = threading.get_ident()
        host = PluginHost([Slow()])
        started = time.perf_counter()
        # another coroutine keeps running while the slow hook waits: the loop is not blocked
        ticks = 0

        async def ticker():
            nonlocal ticks
            while time.perf_counter() - started < 0.15:
                ticks += 1
                await asyncio.sleep(0.01)
        out, _ = await asyncio.gather(host.aon_request(REQ), ticker())
        return out, time.perf_counter() - started, ticks

    out, elapsed, ticks = asyncio.run(run())
    assert out is REQ  # timed out, fail_open -> request unchanged
    assert elapsed < 0.6 and ticks >= 5
    assert loop_thread["hook"] != loop_thread["loop"]


def test_blocking_plugin_fail_closed_raises_on_timeout():
    import asyncio
    import time

    class Strict(Plugin):
        name = "strict"
        blocking = True
        timeout_s = 0.1

        def on_request(self, request):
            time.sleep(0.5)
            return request

    with pytest.raises(TimeoutError):
        asyncio.run(PluginHost([Strict()]).aon_request(REQ))


def test_max_concurrency_queues_calls_outside_the_time_budget():
    import asyncio
    import threading
    import time

    state = {"now": 0, "peak": 0}
    lock = threading.Lock()

    class OneAtATime(Plugin):
        name = "one-at-a-time"
        blocking = True
        timeout_s = 0.15  # each call fits; eight queued calls together would not
        max_concurrency = 2

        def on_request(self, request):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return request

    host = PluginHost([OneAtATime()])

    async def run():
        return await asyncio.gather(*(host.aon_request(REQ) for _ in range(8)))

    assert all(r is REQ for r in asyncio.run(run()))  # no timeouts although 8 x 50 ms > 150 ms
    assert state["peak"] == 2
