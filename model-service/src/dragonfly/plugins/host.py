"""Discovers, loads and runs plugins (see api.py for how to write one)."""

from __future__ import annotations

import asyncio
import logging
import os
from importlib.metadata import entry_points
from typing import Any

from ..schema import DecideRequest
from .api import API_VERSION, Plugin, PluginContext, PluginError, compatible

log = logging.getLogger(__name__)
GROUP = "dragonfly.plugins"


def plugin_config(name: str, environ: dict[str, str] | None = None) -> dict[str, str]:
    """DRAGONFLY_PLUGIN_REDACT_PII_MODE=strict -> {"mode": "strict"} for the plugin named redact-pii."""
    env = os.environ if environ is None else environ
    prefix = f"DRAGONFLY_PLUGIN_{name.upper().replace('-', '_')}_"
    return {k[len(prefix):].lower(): v for k, v in env.items() if k.startswith(prefix)}


def discover(enabled: list[str]) -> list[Plugin]:
    """Instantiate the enabled plugins, in the order listed. Unknown or incompatible names fail startup: a missing
    plugin (say, an auth policy) must never be skipped silently."""
    found = {ep.name: ep for ep in entry_points(group=GROUP)}
    plugins = []
    for name in enabled:
        if name not in found:
            raise RuntimeError(f"plugin {name!r} is enabled but not installed (installed: {sorted(found) or 'none'})")
        cls = found[name].load()
        plugin = cls()
        plugin.name = plugin.name or name
        plugins.append(plugin)
    return plugins


class PluginHost:
    def __init__(self, plugins: list[Plugin], low_confidence: float = 0.0):
        self.plugins = plugins
        self.low_confidence = low_confidence
        for p in plugins:
            p.setup(PluginContext(API_VERSION, plugin_config(p.name)))  # a gRPC plugin learns its api_version here
            if not compatible(p.api_version):
                raise RuntimeError(f"plugin {p.name} targets API {p.api_version}; this host provides {API_VERSION}")
            log.info("plugin loaded: %s %s", p.name, p.version)

    @classmethod
    def from_env(cls) -> PluginHost:
        from .grpc_plugin import from_env as grpc_from_env

        enabled = [n.strip() for n in os.environ.get("DRAGONFLY_PLUGINS", "").split(",") if n.strip()]
        plugins = discover(enabled) + grpc_from_env(os.environ.get("DRAGONFLY_GRPC_PLUGINS", ""))
        return cls(plugins, float(os.environ.get("DRAGONFLY_LOW_CONFIDENCE", "0")))

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": p.name, "version": p.version, "api_version": p.api_version,
                 "kind": "grpc" if hasattr(p, "target") else "python"} for p in self.plugins]

    def _call(self, plugin: Plugin, hook: str, fallback, *args):
        try:
            return getattr(plugin, hook)(*args)
        except PluginError:
            raise
        except Exception:
            if not plugin.fail_open:
                raise
            log.exception("plugin %s failed in %s; skipped (fail_open)", plugin.name, hook)
            return fallback

    def _slots(self, plugin: Plugin) -> asyncio.Semaphore | None:
        if not getattr(plugin, "max_concurrency", None):
            return None
        slots = getattr(self, "_semaphores", None)
        if slots is None:
            slots = self._semaphores = {}
        if id(plugin) not in slots:
            slots[id(plugin)] = asyncio.Semaphore(plugin.max_concurrency)
        return slots[id(plugin)]

    async def _acall(self, plugin: Plugin, hook: str, fallback, *args):
        """Like _call, but a `blocking` plugin runs in a worker thread with its time budget, so a slow LLM or HTTP call
        never stalls the event loop (and with it every other request)."""
        if not plugin.blocking:
            return self._call(plugin, hook, fallback, *args)
        slots = self._slots(plugin)
        try:
            if slots is None:
                return await asyncio.wait_for(asyncio.to_thread(getattr(plugin, hook), *args), timeout=plugin.timeout_s)
            async with slots:  # queueing here does not count against the plugin's time budget
                return await asyncio.wait_for(asyncio.to_thread(getattr(plugin, hook), *args), timeout=plugin.timeout_s)
        except PluginError:
            raise
        except Exception as e:
            if not plugin.fail_open:
                raise
            reason = "timed out" if isinstance(e, TimeoutError) else "failed"
            log.warning("plugin %s %s in %s after %.1fs budget; skipped (fail_open)", plugin.name, reason, hook,
                        plugin.timeout_s)
            return fallback

    async def aon_request(self, request: DecideRequest) -> DecideRequest:
        for p in self.plugins:
            request = await self._acall(p, "on_request", request, request)
        return request

    async def aon_decision(self, request: DecideRequest, response: dict[str, Any]) -> dict[str, Any]:
        if self.low_confidence > 0:
            for qid, answer in list(response["answers"].items()):
                if answer.get("confidence", 1.0) >= self.low_confidence:
                    continue
                for p in self.plugins:
                    replacement = await self._acall(p, "on_low_confidence", None, request, qid, answer)
                    if replacement is not None:
                        response["answers"][qid] = replacement
                        break
        for p in self.plugins:
            response = await self._acall(p, "on_decision", response, request, response)
        return response

    def on_request(self, request: DecideRequest) -> DecideRequest:
        for p in self.plugins:
            request = self._call(p, "on_request", request, request)
        return request

    def on_decision(self, request: DecideRequest, response: dict[str, Any]) -> dict[str, Any]:
        if self.low_confidence > 0:
            for qid, answer in list(response["answers"].items()):
                if answer.get("confidence", 1.0) >= self.low_confidence:
                    continue
                for p in self.plugins:
                    replacement = self._call(p, "on_low_confidence", None, request, qid, answer)
                    if replacement is not None:
                        response["answers"][qid] = replacement
                        break
        for p in self.plugins:
            response = self._call(p, "on_decision", response, request, response)
        return response

    def shutdown(self) -> None:
        for p in self.plugins:
            try:
                p.shutdown()
            except Exception:
                log.exception("plugin %s failed to shut down", p.name)
