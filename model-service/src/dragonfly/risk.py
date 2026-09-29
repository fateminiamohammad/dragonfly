"""Risk-controlled answers: `"max_error": 0.05` in a request.

Calibrated confidence says how sure the model is. Risk control turns that into a promise the caller can build on:

  decided  true when the answer's confidence is above a threshold at which, on held-out data, the error rate of
           accepted answers is at most max_error (a Clopper-Pearson upper bound over ~50 candidate thresholds with a
           Bonferroni correction, so the promise holds with 95% confidence, not just on average).
  set      the smallest set of options that contains the right answer with probability >= 1 - max_error (split
           conformal prediction with the score 1 - p(true answer)). One option = sure; several = let a human or an
           LLM choose among just these.

Both are fitted by scripts/fit_risk.py on the calibration split, on the probabilities the server actually returns
(the cascade's), and stored as JSON (DRAGONFLY_RISK). They hold for data like the calibration data (exchangeability);
the realized error on the test split is reported next to each target so drift is visible.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .schema import question_confidence

TARGETS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
DELTA = 0.05  # the guarantee holds with probability 1 - DELTA over the calibration sample
MIN_PER_TYPE = 200  # fewer calibration questions of a type: use the table for all types


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Binomial(n, p) <= k), summed with the pmf ratio recurrence (exact, no scipy needed)."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    log_term = n * math.log1p(-p)  # log pmf(0)
    ratio = math.log(p) - math.log1p(-p)
    total = 0.0
    for i in range(min(k, n) + 1):
        total += math.exp(log_term)
        if i < n:
            log_term += math.log(n - i) - math.log(i + 1) + ratio
    return min(1.0, total)


def error_upper_bound(errors: int, n: int, delta: float = DELTA) -> float:
    """Clopper-Pearson upper bound: the largest p with P(Binomial(n, p) <= errors) >= delta."""
    if n == 0:
        return 1.0
    if errors >= n:
        return 1.0
    lo, hi = errors / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom_cdf(errors, n, mid) >= delta:
            lo = mid
        else:
            hi = mid
    return hi


def fit_thresholds(conf: list[float], right: list[bool], targets=TARGETS, delta: float = DELTA,
                   candidates: int = 50) -> dict[str, float | None]:
    """Per target error: the lowest confidence threshold whose accepted answers keep the error bound <= target.
    None = no threshold qualifies (never decide at that target).

    Candidate thresholds are the confidence quantiles at 2% steps; each is tested at delta / (number of candidates)
    (Bonferroni), so choosing the best one afterwards still leaves the promise valid at 1 - delta."""
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    n = len(order)
    if n == 0:
        return {str(e): None for e in targets}
    cuts = sorted({max(1, round(n * j / candidates)) for j in range(1, candidates + 1)})
    level = delta / len(cuts)
    # accepted = the top-k answers; extend k over ties so a threshold never splits equal confidences
    stats = []  # (threshold, errors, accepted)
    errors_at = [0] * (n + 1)
    for k, i in enumerate(order, 1):
        errors_at[k] = errors_at[k - 1] + (not right[i])
    for k in cuts:
        while k < n and conf[order[k]] == conf[order[k - 1]]:
            k += 1
        stats.append((conf[order[k - 1]], errors_at[k], k))
    out: dict[str, float | None] = {}
    for eps in targets:
        ok = [t for t, e, k in stats if error_upper_bound(e, k, level) <= eps]
        out[str(eps)] = min(ok) if ok else None
    return out


def fit_set_quantiles(p_true: list[float], targets=TARGETS) -> dict[str, float]:
    """Split conformal: per target error alpha, the threshold on 1 - p(true) (answers with p >= 1 - q are in the set)."""
    scores = sorted(1 - p for p in p_true)
    n = len(scores)
    out = {}
    for alpha in targets:
        rank = math.ceil((n + 1) * (1 - alpha))
        out[str(alpha)] = 1.0 if rank > n else scores[rank - 1]
    return out


def fit(probs: list[list[float]], labels: list[int], qtypes: list[str], targets=TARGETS, delta: float = DELTA) -> dict:
    """probs per question (as served), the true option index, the question type. -> the table fit_risk.py saves."""
    def table(idx):
        conf = [question_confidence(qtypes[i], probs[i]) for i in idx]
        right = [max(range(len(probs[i])), key=probs[i].__getitem__) == labels[i] for i in idx]
        return {"n": len(idx), "decide": fit_thresholds(conf, right, targets, delta),
                "set": fit_set_quantiles([probs[i][labels[i]] for i in idx], targets)}
    everything = list(range(len(probs)))
    tables = {"all": table(everything)}
    for qt in sorted(set(qtypes)):
        idx = [i for i in everything if qtypes[i] == qt]
        if len(idx) >= MIN_PER_TYPE:
            tables[qt] = table(idx)
    return {"targets": list(targets), "delta": delta, "tables": tables}


class RiskTable:
    def __init__(self, data: dict):
        self.data = data
        self.targets = sorted(data["targets"])

    @classmethod
    def load(cls, path: str) -> RiskTable:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def target_for(self, max_error: float) -> float | None:
        """The largest fitted target not above max_error (conservative), or None if max_error is below them all."""
        ok = [t for t in self.targets if t <= max_error + 1e-12]
        return max(ok) if ok else None

    def apply(self, answer: dict, probs: list[float], keys: list[str], qtype: str, max_error: float) -> dict:
        """Adds decided / set / risk to one answer (in place) and returns it."""
        eps = self.target_for(max_error)
        t = self.data["tables"].get(qtype) or self.data["tables"]["all"]
        if eps is None:
            answer.update(decided=False, set=list(keys), risk={"max_error": max_error, "note": "below the smallest "
                          f"fitted target {self.targets[0]}; nothing can be promised"})
            return answer
        threshold = t["decide"].get(str(eps))
        conf = question_confidence(qtype, probs)
        q = t["set"][str(eps)]
        members = [k for k, p in zip(keys, probs) if p >= 1 - q] or [keys[max(range(len(probs)), key=probs.__getitem__)]]
        answer.update(decided=threshold is not None and conf >= threshold,
                      set=sorted(members, key=lambda k: -probs[keys.index(k)]),
                      risk={"max_error": eps, "confidence_level": 1 - self.data["delta"],
                            "threshold": threshold, "calibration_questions": t["n"]})
        return answer
