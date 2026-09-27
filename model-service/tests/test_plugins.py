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
