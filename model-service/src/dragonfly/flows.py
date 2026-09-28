"""Agent flows: chains of dragonflies. Each step asks one model (general, a specialist or "auto") a few questions about
the state; edges pick the next step from the answers. A whole flow runs in-process: no network hop between steps, and
each step is one batched decision (a few ms on tier S), so a 3-step flow is still far faster than one LLM call.

    {
      "name": "support-triage",
      "start": "classify",
      "steps": {
        "classify": {
          "model": "dragonfly-latest",
          "questions": {"intent": {"type": "choice", "criteria": {"refund": null, "delivery": null, "other": null}}},
          "next": [{"if": "intent.choice == refund and intent.confidence > 0.7", "to": "refund_risk"},
                   {"if": "intent.choice == delivery", "to": "delivery"}]
        },
        "refund_risk": {
          "model": "fraud",
          "state": "{{state}}\\n\\nThe customer wants a refund.",
          "questions": {"risky": {"type": "noul", "instructions": "Is this refund request risky?"}}
        },
        "delivery": {"questions": {"late": {"type": "noul", "instructions": "Is the parcel late?"}}}
      }
    }

Conditions: clauses `<question>.<field> <op> <value>` joined by `and`, alternatives by `or` (and binds tighter).
Fields are those of an answer (choice, noul, score, confidence) plus `value` (whichever of choice/noul/score the
question has). The first edge whose condition holds wins; an edge without "if" always holds; no match ends the flow.
A step's "state" is a template: {{state}} is the flow's input, {{<step>.<question>.<field>}} an earlier answer.
"""

from __future__ import annotations

import json
import operator
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

MAX_STEPS = 16
OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq, "!=": operator.ne, ">": operator.gt, "<": operator.lt}
CLAUSE = re.compile(r"^\s*([\w-]+)\.(\w+)\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$")
PLACEHOLDER = re.compile(r"\{\{\s*([\w.-]+)\s*\}\}")
NAME = re.compile(r"^[\w-]{1,64}$")


class FlowError(ValueError):
    pass


def answer_field(answer: dict, field: str):
    if field == "value":
        for k in ("choice", "noul", "score"):
            if k in answer:
                return answer[k]
        return None
    return answer.get(field)


def parse_condition(expr: str | None) -> Callable[[dict], bool]:
    """-> predicate(answers of the current step). None/empty -> always true."""
    if not expr or not expr.strip():
        return lambda answers: True
    alternatives = []
    for alt in re.split(r"\s+or\s+", expr.strip()):
        clauses = []
        for part in re.split(r"\s+and\s+", alt):
            m = CLAUSE.match(part)
            if not m:
                raise FlowError(f"bad condition {part!r}: expected <question>.<field> <op> <value>")
            qid, field, op, raw = m.groups()
            raw = raw.strip("'\"")
            value: Any = raw
            if raw.lower() in ("true", "false"):
                value = raw.lower() == "true"
            else:
                try:
                    value = float(raw)
                except ValueError:
                    pass
            clauses.append((qid, field, OPS[op], value))
        alternatives.append(clauses)

    def holds(answers: dict, clause) -> bool:
        qid, field, op, value = clause
        got = answer_field(answers.get(qid, {}), field)
        if got is None:
            return False
        try:
            if isinstance(value, bool):
                got = got > 0.5 if isinstance(got, float) and field in ("noul", "value") else bool(got)
            elif isinstance(value, float):
                got = float(got)
            else:
                got = str(got)
            return op(got, value)
        except (TypeError, ValueError):
            return False

    return lambda answers: any(all(holds(answers, c) for c in clauses) for clauses in alternatives)


def render_state(template: Any, state: Any, history: dict[str, dict]) -> Any:
    """A step's state: the flow input when the step has none, else the template with placeholders filled in."""
    if template is None:
        return state
    if not isinstance(template, str):
        return template

    def fill(m: re.Match) -> str:
        key = m.group(1)
        if key == "state":
            return state if isinstance(state, str) else str(state)
        parts = key.split(".")
        if len(parts) != 3:
            raise FlowError(f"bad placeholder {{{{{key}}}}}: use {{{{state}}}} or {{{{step.question.field}}}}")
        step, qid, field = parts
        got = answer_field(history.get(step, {}).get(qid, {}), field)
        return "" if got is None else str(got)

    return PLACEHOLDER.sub(fill, template)


def validate(flow: dict) -> dict:
    """Checks a flow definition up front (so a bad edge fails at save time, not in production). Returns it."""
    steps = flow.get("steps")
    if not isinstance(steps, dict) or not steps:
        raise FlowError("a flow needs a non-empty steps object")
    start = flow.get("start") or next(iter(steps))
    if start not in steps:
        raise FlowError(f"start step {start!r} is not defined")
    for name, step in steps.items():
        if not NAME.match(name):
            raise FlowError(f"bad step name {name!r}")
        if not isinstance(step.get("questions"), dict) or not step["questions"]:
            raise FlowError(f"step {name!r} needs questions")
        for edge in step.get("next", []):
            if edge.get("to") not in steps:
                raise FlowError(f"step {name!r} goes to undefined step {edge.get('to')!r}")
            parse_condition(edge.get("if"))
    return {**flow, "start": start}


Decide = Callable[[dict], Awaitable[dict]]


async def run_flow(flow: dict, state: Any, decide: Decide, max_steps: int = MAX_STEPS) -> dict:
    """decide(request body) -> response body (the /v1/systemone pipeline). Returns the trace of the run."""
    flow = validate(flow)
    started = time.perf_counter()
    history: dict[str, dict] = {}
    trace = []
    current = flow["start"]
    while current is not None:
        if len(trace) >= max_steps:
            raise FlowError(f"flow did not finish within {max_steps} steps (a loop?)")
        step = flow["steps"][current]
        body = {"state": render_state(step.get("state"), state, history), "questions": step["questions"]}
        if step.get("model"):
            body["model"] = step["model"]
        t = time.perf_counter()
        response = await decide(body)
        history[current] = response["answers"]
        nxt = next((e["to"] for e in step.get("next", []) if parse_condition(e.get("if"))(response["answers"])), None)
        trace.append({"step": current, "model": step.get("model", "dragonfly-latest"),
                      "specialist": response.get("specialist"), "answers": response["answers"],
                      "latency_ms": round((time.perf_counter() - t) * 1000, 2),
                      "model_ms": response.get("latency_ms"), "next": nxt})
        current = nxt
    return {"flow": flow.get("name"), "path": [s["step"] for s in trace], "steps": trace, "answers": history,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2)}


def mermaid(flow: dict, path: list[str] | None = None) -> str:
    """The flow as a Mermaid flowchart; steps on `path` (a run's path) are highlighted."""
    flow = validate(flow)
    lines = ["flowchart LR"]
    for name, step in flow["steps"].items():
        qs = ", ".join(step["questions"])
        label = f"{name}<br/><small>{step.get('model', 'general')}: {qs}</small>"
        lines.append(f'    {name}["{label}"]')
    for name, step in flow["steps"].items():
        for edge in step.get("next", []):
            cond = (edge.get("if") or "otherwise").replace('"', "'")
            lines.append(f'    {name} -- "{cond}" --> {edge["to"]}')
    if path:
        lines.append("    classDef taken fill:#1d6b5a,stroke:#3fd0a8,color:#fff")
        lines.append(f"    class {','.join(dict.fromkeys(path))} taken")
    return "\n".join(lines)


class RedisFlowStore:
    """Saved flows live in the Redis hash dragonfly:flows (name -> JSON), written by the backend (/api/flows)."""

    KEY = "dragonfly:flows"

    def __init__(self, redis):
        self.redis = redis

    async def get(self, name: str) -> dict | None:
        raw = await self.redis.hget(self.KEY, name)
        return None if raw is None else json.loads(raw)
