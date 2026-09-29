"""Request and response shapes for POST /v1/systemone (TypeSafe / Kev compatible) and POST /v1/decide.

Every question type reduces to one primitive, a pointer over options:

  noul   -> options ["no", "yes"];           answer = p(yes)
  choice -> options are the criteria names;  answer = argmax, probabilities by name, confidence
  score  -> options are the ordered levels;  answer = expected level, probabilities by level index, confidence
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255  # per forward pass
MAX_OPTIONS_TWO_STAGE = 10_000  # choice questions above MAX_OPTIONS: pruned first, then decided (twostage.py)
MODEL_NAMES = ("dragonfly-latest", "jev-latest", "kev-latest")  # SDK defaults, so an unconfigured client works


class Noul(BaseModel):
    type: Literal["noul"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None  # optional descriptions under "false" / "true"


class Choice(BaseModel):
    type: Literal["choice"]
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        # above MAX_OPTIONS the question is answered in two stages (dragonfly/twostage.py)
        if not 1 <= len(self.criteria) <= MAX_OPTIONS_TWO_STAGE:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS_TWO_STAGE} options")
        return self


class Score(BaseModel):
    type: Literal["score"]
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=1, max_length=MAX_OPTIONS)


Question = Union[Noul, Choice, Score]


class DecideRequest(BaseModel):
    state: JSONContent
    model: str = "dragonfly-latest"
    questions: dict[str, Question] = Field(min_length=1)
    # Dragonfly extension (docs/API.md#risk-controlled-answers): each answer then says whether it is "decided" at this
    # error rate, and the smallest option "set" that contains the truth with probability >= 1 - max_error
    max_error: float | None = Field(default=None, gt=0, lt=0.5)


def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten a JSON value into the text the model reads. Object keys are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
        for k, x in v.items()
    )


def option_text(name: str, desc: JSONContent) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_keys(qtype: str, criteria) -> list[str]:
    """Keys the probabilities are reported under, in option order. Labels and soft targets use the same keys."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def to_record(req: DecideRequest) -> tuple[dict, list[dict]]:
    """Request -> internal record {"state", "questions": [{"instr", "options", "qtype"}]} plus metadata to map the
    probabilities back to answers."""
    questions, meta = [], []
    for qid, q in req.questions.items():
        m = {"id": qid, "type": q.type, "keys": question_keys(q.type, q.criteria)}
        if q.type == "noul":
            c = q.criteria or {}
            options = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif q.type == "choice":
            options = [option_text(k, v) for k, v in q.criteria.items()]
        else:
            options = [render(x) for x in q.criteria]
            m["legend"] = dict(zip(m["keys"], options))
        questions.append({"instr": render(q.instructions), "options": options, "qtype": q.type})
        meta.append(m)
    return {"state": render(req.state), "questions": questions}, meta


# Confidence formulas match TypeSafe's reference adapter, so scores are comparable across Jev, Kev and Dragonfly.
def _normalize(p: list[float]) -> list[float]:
    t = sum(p)
    return [1 / len(p)] * len(p) if t == 0 else [x / t for x in p]


def choice_confidence(p: list[float]) -> float:
    """(p_max - 1/K) / (1 - 1/K): 0 at uniform, 1 at certainty."""
    k = len(p)
    return 1.0 if k == 1 else (max(_normalize(p)) - 1 / k) / (1 - 1 / k)


def noul_confidence(p_true: float) -> float:
    return abs(2 * p_true - 1)


def score_confidence(p: list[float]) -> float:
    """1 - E|level - mode| / D, where D is that spread for a uniform distribution. 1 when all mass is on one level."""
    n = len(p)
    if n == 1:
        return 1.0
    p = _normalize(p)
    mode = max(range(n), key=p.__getitem__)
    d = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / d)


def question_confidence(qtype: str, p: list[float]) -> float:
    if qtype == "noul":
        return noul_confidence(p[1])
    if qtype == "choice":
        return choice_confidence(p)
    return score_confidence(p)


def round_prob(x: float) -> float:
    """4 decimals keeps a rounded 255-option distribution within |sum - 1| < 0.02."""
    return round(float(x), 4)


def to_answers(probs: list[list[float]], meta: list[dict]) -> dict[str, Any]:
    out = {}
    for p, m in zip(probs, meta):
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": round_prob(p[1]), "confidence": round_prob(noul_confidence(p[1]))}
        elif m["type"] == "choice":
            best = max(range(len(p)), key=p.__getitem__)
            out[m["id"]] = {
                "type": "choice",
                "choice": m["keys"][best],
                "confidence": round_prob(choice_confidence(p)),
                "probabilities": {k: round_prob(v) for k, v in zip(m["keys"], p)},
            }
        else:
            out[m["id"]] = {
                "type": "score",
                "score": round_prob(sum(i * pi for i, pi in enumerate(p))),
                "legend": m["legend"],
                "probabilities": {str(i): round_prob(v) for i, v in enumerate(p)},
                "confidence": round_prob(score_confidence(p)),
            }
    return out
