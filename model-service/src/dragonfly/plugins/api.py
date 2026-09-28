"""The stable plugin API. Plugins may be open or closed source: they are separate packages that depend only on this
module, so the Apache-2.0 core never has to contain them.

A plugin is a Python package that exposes a `Plugin` subclass under the entry-point group `dragonfly.plugins`:

    # pyproject.toml of your plugin
    [project.entry-points."dragonfly.plugins"]
    redact-pii = "my_plugin:RedactPII"

and is enabled by name with DRAGONFLY_PLUGINS=redact-pii (comma separated; nothing loads unless listed).

Compatibility: API_VERSION follows semver. The host loads a plugin when the major versions match and the plugin's
minor version is not newer than the host's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schema import DecideRequest

API_VERSION = "1.1"  # 1.1: `blocking` / `timeout_s` (additive; 1.0 plugins still load)


@dataclass
class PluginContext:
    """What the host gives a plugin at setup."""

    api_version: str
    config: dict[str, str] = field(default_factory=dict)  # env vars prefixed DRAGONFLY_PLUGIN_<NAME>_


class PluginError(Exception):
    """Raise to reject a request with a 4xx status (e.g. a policy violation); `status` defaults to 422."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


class Plugin:
    """Override any hook; the defaults pass everything through unchanged.

    Hooks run on the request path. A fast, pure-Python hook runs inline. A hook that waits on I/O (an LLM, an HTTP
    call, a database) must set `blocking = True`: the host then runs it in a worker thread, off the event loop that
    serves every other request, and gives up after `timeout_s` seconds (treated like an exception).

    `fail_open = True` means an unexpected exception or timeout is logged and the plugin is skipped for that request;
    `False` (default) fails the request with 500. PluginError always rejects.
    """

    name: str = ""
    version: str = "0.0.0"
    api_version: str = API_VERSION
    fail_open: bool = False
    blocking: bool = False
    timeout_s: float = 5.0
    # blocking plugins: at most this many hook calls in flight (None = unbounded). Extra calls wait their turn before
    # the time budget starts, so a service that handles calls one by one is not pushed past its deadline under load.
    max_concurrency: int | None = None

    def setup(self, ctx: PluginContext) -> None:
        """Called once at startup."""

    def on_request(self, request: DecideRequest) -> DecideRequest:
        """Before the model: validate, redact, enrich the state, add or rewrite questions."""
        return request

    def on_decision(self, request: DecideRequest, response: dict[str, Any]) -> dict[str, Any]:
        """After the model: business rules, overrides, audit logging. `response` is the full response body."""
        return response

    def on_low_confidence(self, request: DecideRequest, question_id: str, answer: dict[str, Any]) -> dict[str, Any] | None:
        """Called for each answer whose confidence is below DRAGONFLY_LOW_CONFIDENCE. Return a replacement answer
        (e.g. from an LLM or a human queue) or None to keep the model's."""
        return None

    def shutdown(self) -> None:
        """Called once at shutdown."""


def compatible(plugin_version: str, host_version: str = API_VERSION) -> bool:
    p_major, p_minor = (int(x) for x in plugin_version.split(".")[:2])
    h_major, h_minor = (int(x) for x in host_version.split(".")[:2])
    return p_major == h_major and p_minor <= h_minor
