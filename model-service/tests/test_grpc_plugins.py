"""Out-of-process plugins: a real gRPC server in-process, and byte compatibility with protoc-generated code."""

import json
import subprocess
import sys
import time
from concurrent import futures
from pathlib import Path

import grpc
import pytest

from dragonfly.plugins import PluginError, PluginHost, wire
from dragonfly.plugins.grpc_plugin import GrpcPlugin, from_env
from dragonfly.schema import DecideRequest

PROTO_ROOT = Path(__file__).resolve().parents[2] / "proto"
REQ = DecideRequest.model_validate({"state": "hello", "questions": {"q": {"type": "noul"}}})


def handler(slow: bool = False):
    """A policy plugin written against the wire codec: rewrites the state, flags low confidence, rejects 'forbidden'."""

    def describe(raw, ctx):
        return wire.encode("PluginInfo", {"name": "policy", "version": "1.2.3", "api_version": "1.0",
                                          "hooks": ["on_request", "on_decision", "on_low_confidence"]})

    def on_request(raw, ctx):
        if slow:
            time.sleep(0.5)
        req = json.loads(wire.decode("HookRequest", raw)["request_json"])
        if req["state"] == "forbidden":
            return wire.encode("HookReply", {"reject_status": 403, "reject_message": "not allowed"})
        req["state"] = req["state"].upper()
        return wire.encode("HookReply", {"request_json": json.dumps(req)})

    def on_decision(raw, ctx):
        resp = json.loads(wire.decode("HookRequest", raw)["response_json"])
        resp["reviewed"] = True
        return wire.encode("HookReply", {"response_json": json.dumps(resp)})

    def on_low(raw, ctx):
        msg = wire.decode("HookRequest", raw)
        answer = {**json.loads(msg["answer_json"]), "noul": 1.0, "by": msg["question_id"]}
        return wire.encode("HookReply", {"replaced": True, "answer_json": json.dumps(answer)})

    methods = {"Describe": describe, "OnRequest": on_request, "OnDecision": on_decision, "OnLowConfidence": on_low}
    return grpc.method_handlers_generic_handler("dragonfly.plugin.v1.PluginService", {
        name: grpc.unary_unary_rpc_method_handler(fn) for name, fn in methods.items()})


@pytest.fixture
def server():
    started = []

    def start(slow: bool = False) -> str:
        s = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        s.add_generic_rpc_handlers((handler(slow),))
        port = s.add_insecure_port("127.0.0.1:0")
        s.start()
        started.append(s)
        return f"127.0.0.1:{port}"

    yield start
    for s in started:
        s.stop(0)


def test_hooks_over_grpc(server, monkeypatch):
    monkeypatch.setenv("DRAGONFLY_PLUGIN_POLICY_TIMEOUT_MS", "2000")
    host = PluginHost([GrpcPlugin("policy", server())], low_confidence=0.5)
    assert host.describe() == [{"name": "policy", "version": "1.2.3", "api_version": "1.0", "kind": "grpc"}]
    assert host.on_request(REQ).state == "HELLO"
    resp = host.on_decision(REQ, {"answers": {"q": {"type": "noul", "noul": 0.5, "confidence": 0.0}}})
    assert resp["reviewed"] is True
    assert resp["answers"]["q"] == {"type": "noul", "noul": 1.0, "confidence": 0.0, "by": "q"}


def test_rejection_becomes_http_status(server):
    host = PluginHost([GrpcPlugin("policy", server())])
    with pytest.raises(PluginError) as e:
        host.on_request(REQ.model_copy(update={"state": "forbidden"}))
    assert e.value.status == 403


def test_timeout_fails_closed_by_default_and_open_when_configured(server, monkeypatch):
    target = server(slow=True)
    monkeypatch.setenv("DRAGONFLY_PLUGIN_POLICY_TIMEOUT_MS", "50")
    with pytest.raises(grpc.RpcError):
        PluginHost([GrpcPlugin("policy", target)]).on_request(REQ)
    monkeypatch.setenv("DRAGONFLY_PLUGIN_POLICY_FAIL_OPEN", "1")
    assert PluginHost([GrpcPlugin("policy", target)]).on_request(REQ) is REQ


def test_env_spec_parsing():
    plugins = from_env("a@host-a:1, b@host-b:2")
    assert [(p.name, p.target) for p in plugins] == [("a", "host-a:1"), ("b", "host-b:2")]
    with pytest.raises(RuntimeError):
        from_env("no-target")


def test_wire_codec_matches_protoc(tmp_path):
    """The hand-written codec must produce the same bytes as protoc-generated protobuf code."""
    pytest.importorskip("grpc_tools")
    pytest.importorskip("google.protobuf")
    subprocess.run([sys.executable, "-m", "grpc_tools.protoc", f"-I{PROTO_ROOT}", f"--python_out={tmp_path}",
                    "dragonfly/plugin/v1/plugin.proto"], check=True)
    sys.path.insert(0, str(tmp_path / "dragonfly" / "plugin" / "v1"))
    try:
        import plugin_pb2 as pb
    finally:
        sys.path.pop(0)
    info = {"name": "p", "version": "1", "api_version": "1.0", "hooks": ["on_request", "on_decision"]}
    assert wire.encode("PluginInfo", info) == pb.PluginInfo(**info).SerializeToString()
    reply = {"response_json": '{"a": "ü"}', "replaced": True, "reject_status": 403, "reject_message": "no"}
    assert wire.encode("HookReply", reply) == pb.HookReply(**reply).SerializeToString()
    assert wire.decode("HookReply", pb.HookReply(**reply).SerializeToString()) == {
        "request_json": "", "answer_json": "", **reply}
