"""webhook-audit: send decisions to a webhook, Slack or a JSONL file, without slowing the decision down.

on_decision only puts the event on an in-memory queue and returns; a background thread delivers it (with retries and
exponential backoff). A full queue drops the oldest-unsent event and counts it, rather than blocking requests.

  DRAGONFLY_PLUGINS=webhook-audit
  DRAGONFLY_PLUGIN_WEBHOOK_AUDIT_URL=https://example.com/hooks/dragonfly   (JSON POST of the event)
  DRAGONFLY_PLUGIN_WEBHOOK_AUDIT_SLACK_URL=https://hooks.slack.com/...     (a one-line summary)
  DRAGONFLY_PLUGIN_WEBHOOK_AUDIT_FILE=/runs/audit.jsonl                    (append one JSON line per event)
  DRAGONFLY_PLUGIN_WEBHOOK_AUDIT_WHEN=escalate.noul>0.8                    (optional filter; default: every decision)
  DRAGONFLY_PLUGIN_WEBHOOK_AUDIT_INCLUDE_STATE=0                           (1 to include the document; off for privacy)

Filters: "<question>.<field><op><value>" joined by " or ", op one of > >= < <= == !=, e.g.
  "intent.choice==refund or risk.score>=3"
"""

from __future__ import annotations

import json
import logging
import operator
import queue
import re
import threading
import time
import urllib.request
from datetime import UTC, datetime
from typing import Any

from ..plugins import Plugin, PluginContext
from ..schema import DecideRequest

log = logging.getLogger(__name__)
OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq, "!=": operator.ne, ">": operator.gt, "<": operator.lt}
CONDITION = re.compile(r"^\s*([\w-]+)\.(\w+)\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$")


def parse_filter(expr: str):
    """-> predicate(response) -> bool. Empty -> always True. Raises ValueError on a malformed filter (at startup)."""
    if not expr.strip():
        return lambda response: True
    clauses = []
    for part in expr.split(" or "):
        m = CONDITION.match(part)
        if not m:
            raise ValueError(f"bad filter clause {part!r}; expected <question>.<field><op><value>")
        qid, field, op, raw = m.groups()
        try:
            value: Any = float(raw)
        except ValueError:
            value = raw.strip("'\"")
        clauses.append((qid, field, OPS[op], value))

    def predicate(response: dict) -> bool:
        for qid, field, op, value in clauses:
            got = response.get("answers", {}).get(qid, {}).get(field)
            if got is None:
                continue
            try:
                if op(float(got) if isinstance(value, float) else str(got), value):
                    return True
            except (TypeError, ValueError):
                continue
        return False
    return predicate


def slack_text(event: dict) -> str:
    parts = []
    for qid, a in event["answers"].items():
        value = a.get("choice", a.get("noul", a.get("score")))
        parts.append(f"*{qid}*: {value} ({a.get('confidence', 0):.0%})")
    return "Dragonfly decision · " + " · ".join(parts)


class WebhookAudit(Plugin):
    name = "webhook-audit"
    version = "1.0.0"
    api_version = "1.1"
    fail_open = True  # auditing must never fail a decision

    def setup(self, ctx: PluginContext) -> None:
        self.url = ctx.config.get("url")
        self.slack_url = ctx.config.get("slack_url")
        self.file = ctx.config.get("file")
        self.include_state = ctx.config.get("include_state", "0") == "1"
        self.when = parse_filter(ctx.config.get("when", ""))
        self.retries = int(ctx.config.get("retries", "3"))
        self.queue: queue.Queue = queue.Queue(maxsize=int(ctx.config.get("queue_size", "10000")))
        self.sent = self.failed = self.dropped = 0
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._deliver_loop, name="webhook-audit", daemon=True)
        self.thread.start()

    def on_decision(self, request: DecideRequest, response: dict[str, Any]) -> dict[str, Any]:
        if not self.when(response):
            return response
        event = {"time": datetime.now(UTC).isoformat(), "model": response.get("model"), "answers": response["answers"],
                 "latency_ms": response.get("latency_ms"), "usage": response.get("usage")}
        if self.include_state:
            event["state"] = request.state
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1
        return response

    def _post(self, url: str, payload: dict) -> None:
        req = urllib.request.Request(url, data=json.dumps(payload, default=str).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            if r.status >= 300:
                raise RuntimeError(f"HTTP {r.status}")

    def deliver(self, event: dict) -> None:
        """One event to every configured target, with retries (exponential backoff) per target."""
        targets = []
        if self.url:
            targets.append(lambda: self._post(self.url, event))
        if self.slack_url:
            targets.append(lambda: self._post(self.slack_url, {"text": slack_text(event)}))
        if self.file:
            def write():
                with open(self.file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
            targets.append(write)
        for send in targets:
            for attempt in range(self.retries + 1):
                try:
                    send()
                    self.sent += 1
                    break
                except Exception as e:
                    if attempt == self.retries:
                        self.failed += 1
                        log.warning("webhook-audit: delivery failed after %d attempts: %s", attempt + 1, e)
                    else:
                        time.sleep(min(2 ** attempt * 0.5, 10))

    def _deliver_loop(self) -> None:
        while not self.stopping.is_set():
            try:
                event = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.deliver(event)

    def shutdown(self) -> None:
        deadline = time.time() + 5  # flush what we can
        while not self.queue.empty() and time.time() < deadline:
            time.sleep(0.05)
        self.stopping.set()
