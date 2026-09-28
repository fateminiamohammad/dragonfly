"""guardrails: safety checks on every request before the model sees it.

  DRAGONFLY_PLUGINS=guardrails
  DRAGONFLY_PLUGIN_GUARDRAILS_PII=redact            redact | off              (emails, card numbers, IBANs, phones)
  DRAGONFLY_PLUGIN_GUARDRAILS_INJECTION=flag        block | flag | off        (prompt-injection phrases)
  DRAGONFLY_PLUGIN_GUARDRAILS_PROFANITY=flag        block | flag | off
  DRAGONFLY_PLUGIN_GUARDRAILS_MAX_STATE_CHARS=100000
  DRAGONFLY_PLUGIN_GUARDRAILS_PROFANITY_WORDS=word1,word2   (extra words; a small English default list is built in)

"block" rejects the request (HTTP 422, the reason in the message). "flag" lets it through and adds
"guardrails": {"flags": [...], "redacted": n} to the response, so the caller can decide.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from ..plugins import Plugin, PluginContext, PluginError
from ..schema import DecideRequest

PII = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b"),
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "phone": re.compile(r"(?<!\w)\+?\d[\d ()-]{7,}\d"),
}
INJECTION = re.compile("|".join([
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the |all )?(previous|prior|above|system)",
    r"you are now (?:a|an|in) ",
    r"system prompt",
    r"answer (?:with|only) ['\"]?(?:yes|true|approve)",
    r"(?:reveal|print|show) (?:your|the) (?:instructions|prompt)",
    r"override (?:the )?(?:rules|policy|decision)",
]), re.IGNORECASE)
PROFANITY = {"fuck", "fucking", "shit", "bitch", "asshole", "bastard", "dick", "cunt"}


def walk_strings(value, fn):
    """Apply fn to every string inside a JSON value; returns the new value."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [walk_strings(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: walk_strings(v, fn) for k, v in value.items()}
    return value


class Guardrails(Plugin):
    name = "guardrails"
    version = "1.0.0"
    api_version = "1.1"

    def setup(self, ctx: PluginContext) -> None:
        self.pii = ctx.config.get("pii", "redact")
        self.injection = ctx.config.get("injection", "flag")
        self.profanity = ctx.config.get("profanity", "flag")
        self.max_chars = int(ctx.config.get("max_state_chars", "100000"))
        self.words = PROFANITY | {w.strip().lower() for w in ctx.config.get("profanity_words", "").split(",") if w.strip()}
        self._flags: dict[int, dict] = {}  # id(request) -> findings, handed from on_request to on_decision
        self._lock = threading.Lock()

    def check(self, text: str) -> list[str]:
        flags = []
        if self.injection != "off" and INJECTION.search(text):
            flags.append("prompt_injection")
        if self.profanity != "off" and any(w in self.words for w in re.findall(r"[a-z']+", text.lower())):
            flags.append("profanity")
        return flags

    def on_request(self, request: DecideRequest) -> DecideRequest:
        text = request.state if isinstance(request.state, str) else json.dumps(request.state, ensure_ascii=False)
        if len(text) > self.max_chars:
            raise PluginError(f"guardrails: state is {len(text)} characters; the limit is {self.max_chars}", 413)
        flags = self.check(text)
        for flag in flags:
            mode = self.injection if flag == "prompt_injection" else self.profanity
            if mode == "block":
                raise PluginError(f"guardrails: request blocked ({flag.replace('_', ' ')} detected)", 422)
        redacted = 0
        if self.pii == "redact":
            def redact(s: str) -> str:
                nonlocal redacted
                for kind, pattern in PII.items():
                    s, n = pattern.subn(f"[{kind}]", s)
                    redacted += n
                return s
            request = request.model_copy(update={"state": walk_strings(request.state, redact)})
        if flags or redacted:
            with self._lock:
                self._flags[id(request)] = {"flags": flags, "redacted": redacted}
                while len(self._flags) > 10_000:  # requests rejected later never reach on_decision: stay bounded
                    self._flags.pop(next(iter(self._flags)))
        return request

    def on_decision(self, request: DecideRequest, response: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            found = self._flags.pop(id(request), None)
        if found:
            response["guardrails"] = found
        return response
