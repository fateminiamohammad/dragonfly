"""llm-escalation: when Dragonfly is unsure, ask an LLM instead.

Runs on on_low_confidence, i.e. only for answers below DRAGONFLY_LOW_CONFIDENCE, so the LLM's cost and latency are
paid only for the hard minority. The LLM is constrained to the question's options and must answer with one of them;
anything else keeps Dragonfly's own answer.

  DRAGONFLY_PLUGINS=llm-escalation
  DRAGONFLY_LOW_CONFIDENCE=0.3
  DRAGONFLY_PLUGIN_LLM_ESCALATION_BASE_URL=http://llm:11434/v1     (default: the local Ollama in the llm profile)
  DRAGONFLY_PLUGIN_LLM_ESCALATION_MODEL=qwen2.5:7b-instruct
  DRAGONFLY_PLUGIN_LLM_ESCALATION_API_KEY=...                      (for hosted APIs)
  DRAGONFLY_PLUGIN_LLM_ESCALATION_TIMEOUT_S=8

The replacement answer keeps the API shape, with "tier": "llm" and "calibrated": false: an LLM's pick is not a
calibrated probability, so its probabilities are one-hot and its confidence is the model's own claim (1.0).
"""

from __future__ import annotations

import json
import re
import urllib.request

from ..plugins import Plugin, PluginContext
from ..schema import DecideRequest, question_keys, render


def build_prompt(request: DecideRequest, qid: str) -> tuple[str, list[str]]:
    q = request.questions[qid]
    keys = question_keys(q.type, q.criteria)
    lines = [f"Document:\n{render(request.state)}", ""]
    if q.instructions:
        lines.append(f"Question: {render(q.instructions)}")
    if q.type == "choice":
        options = [k if not v else f"{k}: {render(v)}" for k, v in q.criteria.items()]
    elif q.type == "noul":
        options = ["true", "false"]
    else:
        options = [f"{i}: {render(c)}" for i, c in enumerate(q.criteria)]
    lines.append("Options:\n" + "\n".join(f"- {o}" for o in options))
    lines.append('Reply with only JSON: {"answer": "<one option key>"}')
    return "\n".join(lines), keys


def parse_answer(text: str, keys: list[str]) -> str | None:
    for block in reversed(re.findall(r"\{[^{}]*\}", text or "")):
        try:
            value = str(json.loads(block).get("answer", "")).strip().lower()
        except (ValueError, AttributeError):
            continue
        for k in keys:
            if value == k.lower() or value.split(":")[0].strip() == k.lower():
                return k
    return None


def as_answer(qtype: str, keys: list[str], pick: str, legend: dict | None) -> dict:
    probs = {k: float(k == pick) for k in keys}
    base = {"type": qtype, "confidence": 1.0, "tier": "llm", "calibrated": False}
    if qtype == "noul":
        return {**base, "noul": probs["true"]}
    if qtype == "choice":
        return {**base, "choice": pick, "probabilities": probs}
    return {**base, "score": float(pick), "probabilities": probs, "legend": legend or {}}


class LLMEscalation(Plugin):
    name = "llm-escalation"
    version = "1.0.0"
    api_version = "1.1"
    blocking = True  # an HTTP call to an LLM: never on the event loop
    fail_open = True  # an unreachable LLM must not fail the decision: keep Dragonfly's answer

    def setup(self, ctx: PluginContext) -> None:
        self.base_url = ctx.config.get("base_url", "http://llm:11434/v1").rstrip("/")
        self.model = ctx.config.get("model", "qwen2.5:7b-instruct")
        self.api_key = ctx.config.get("api_key", "ollama")
        self.timeout_s = float(ctx.config.get("timeout_s", "8"))

    def ask(self, prompt: str) -> str:
        body = {"model": self.model, "temperature": 0, "max_tokens": 32,
                "messages": [{"role": "system", "content": "You answer typed questions about a document. Output only JSON."},
                             {"role": "user", "content": prompt}]}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json", "authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]

    def on_low_confidence(self, request: DecideRequest, question_id: str, answer: dict) -> dict | None:
        prompt, keys = build_prompt(request, question_id)
        pick = parse_answer(self.ask(prompt), keys)
        if pick is None:
            return None  # the LLM didn't give a valid option: keep Dragonfly's answer
        return as_answer(answer["type"], keys, pick, answer.get("legend")) | {"escalated_from": answer}
