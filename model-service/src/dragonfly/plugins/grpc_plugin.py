"""Out-of-process plugins over gRPC (proto/dragonfly/plugin/v1/plugin.proto). A sidecar container, in any language
and open or closed source, implements the hooks; the host calls it with a per-plugin time budget.

  DRAGONFLY_GRPC_PLUGINS=fraud-rules@fraud-rules:50051,audit@audit:50051
  DRAGONFLY_PLUGIN_FRAUD_RULES_TIMEOUT_MS=20     time budget per call (default 50)
  DRAGONFLY_PLUGIN_FRAUD_RULES_FAIL_OPEN=1       on timeout or error: skip the plugin (default: fail the request)
  DRAGONFLY_PLUGIN_FRAUD_RULES_MAX_CONCURRENCY=8  calls in flight at once; the rest queue before their deadline starts (default 4)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..schema import DecideRequest
from . import wire
from .api import Plugin, PluginContext, PluginError

log = logging.getLogger(__name__)
SERVICE = "/dragonfly.plugin.v1.PluginService/"


class GrpcPlugin(Plugin):
    def __init__(self, name: str, target: str):
        self.name = name
        self.target = target
        self.version = "?"
        self.hooks: set[str] = set()

    def setup(self, ctx: PluginContext) -> None:
        import grpc  # optional dependency: pip install "dragonfly-model[grpc]"

        self.timeout = float(ctx.config.get("timeout_ms", "50")) / 1000
        self.fail_open = ctx.config.get("fail_open", "0") in ("1", "true", "yes")
        self.blocking = True  # a network call: run off the event loop
        self.timeout_s = self.timeout + 1.0  # the gRPC deadline fires first; this is only a backstop
        # calls in flight at once: a sidecar that handles them one by one (e.g. Python) stays within its deadline
        self.max_concurrency = int(ctx.config.get("max_concurrency", "4")) or None
        self.channel = grpc.insecure_channel(self.target)
        self._rpc = {
            m: self.channel.unary_unary(SERVICE + m, request_serializer=lambda b: b, response_deserializer=lambda b: b)
            for m in ("Describe", "OnRequest", "OnDecision", "OnLowConfidence")
        }
        info = self._describe(ctx.api_version)
        self.version = info["version"] or "?"
        self.api_version = info["api_version"] or ctx.api_version
        self.hooks = set(info["hooks"])
        log.info("grpc plugin %s at %s: version %s, hooks %s", self.name, self.target, self.version, sorted(self.hooks))

    def _describe(self, api_version: str, attempts: int = 30) -> dict:
        """The sidecar may start after the model-service: retry for about 30 s before failing startup."""
        payload = wire.encode("DescribeRequest", {"host_api_version": api_version})
        for i in range(attempts):
            try:
                return wire.decode("PluginInfo", self._rpc["Describe"](payload, timeout=2))
            except Exception as e:
                if i == attempts - 1:
                    raise RuntimeError(f"grpc plugin {self.name} at {self.target} is unreachable: {e}") from e
                time.sleep(1)
        raise AssertionError("unreachable")

    def _call(self, method: str, values: dict) -> dict:
        reply = wire.decode("HookReply", self._rpc[method](wire.encode("HookRequest", values), timeout=self.timeout))
        if reply["reject_status"] > 0:
            raise PluginError(reply["reject_message"] or f"rejected by plugin {self.name}", reply["reject_status"])
        return reply

    def on_request(self, request: DecideRequest) -> DecideRequest:
        if "on_request" not in self.hooks:
            return request
        reply = self._call("OnRequest", {"request_json": request.model_dump_json()})
        return DecideRequest.model_validate_json(reply["request_json"]) if reply["request_json"] else request

    def on_decision(self, request: DecideRequest, response: dict[str, Any]) -> dict[str, Any]:
        if "on_decision" not in self.hooks:
            return response
        reply = self._call("OnDecision", {"request_json": request.model_dump_json(), "response_json": json.dumps(response)})
        return json.loads(reply["response_json"]) if reply["response_json"] else response

    def on_low_confidence(self, request: DecideRequest, question_id: str, answer: dict[str, Any]) -> dict[str, Any] | None:
        if "on_low_confidence" not in self.hooks:
            return None
        reply = self._call("OnLowConfidence", {"request_json": request.model_dump_json(), "question_id": question_id,
                                                "answer_json": json.dumps(answer)})
        return json.loads(reply["answer_json"]) if reply["replaced"] else None

    def shutdown(self) -> None:
        if getattr(self, "channel", None) is not None:
            self.channel.close()


def from_env(spec: str) -> list[GrpcPlugin]:
    """"a@host:port,b@host:port" -> plugins, in order."""
    plugins = []
    for item in (s.strip() for s in spec.split(",")):
        if not item:
            continue
        if "@" not in item:
            raise RuntimeError(f"DRAGONFLY_GRPC_PLUGINS entry {item!r} must look like name@host:port")
        name, target = item.split("@", 1)
        plugins.append(GrpcPlugin(name, target))
    return plugins
