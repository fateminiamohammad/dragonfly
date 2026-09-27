"""Masks PII in the state before the model sees it. A template for writing your own (open or closed source) plugin.

Settings: DRAGONFLY_PLUGIN_REDACT_PII_MASK (default "[redacted]").
"""

import re

from dragonfly.plugins import Plugin, PluginContext
from dragonfly.schema import DecideRequest

PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # email
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),  # card number
    re.compile(r"\+?\d[\d ()-]{7,}\d"),  # phone
]


def redact(value, mask: str):
    if isinstance(value, str):
        for p in PATTERNS:
            value = p.sub(mask, value)
        return value
    if isinstance(value, list):
        return [redact(v, mask) for v in value]
    if isinstance(value, dict):
        return {k: redact(v, mask) for k, v in value.items()}
    return value


class RedactPII(Plugin):
    name = "redact-pii"
    version = "0.1.0"
    api_version = "1.0"

    def setup(self, ctx: PluginContext) -> None:
        self.mask = ctx.config.get("mask", "[redacted]")

    def on_request(self, request: DecideRequest) -> DecideRequest:
        return request.model_copy(update={"state": redact(request.state, self.mask)})
