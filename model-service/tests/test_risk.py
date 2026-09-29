"""Risk-controlled answers: the guarantee holds on held-out data, sets cover the truth, the API reports both."""

import random

import pytest
from fastapi.testclient import TestClient

from dragonfly.api import create_app
from dragonfly.batching import Worker
from dragonfly.risk import RiskTable, binom_cdf, error_upper_bound, fit, fit_set_quantiles, fit_thresholds


def synthetic(n, seed):
    """A calibrated 3-option model: the top option is right with probability equal to its probability."""
    rng = random.Random(seed)
    probs, labels = [], []
    for _ in range(n):
        top = rng.uniform(0.34, 0.999)
        rest = (1 - top) / 2
        p = [top, rest, rest]
        probs.append(p)
        labels.append(0 if rng.random() < top else rng.choice([1, 2]))
    return probs, labels, ["choice"] * n


def test_binomial_helpers():
    assert binom_cdf(0, 10, 0.1) == pytest.approx(0.9 ** 10)
    assert binom_cdf(10, 10, 0.3) == pytest.approx(1.0)
    ub = error_upper_bound(5, 100)
    assert 0.05 < ub < 0.12  # 5 errors in 100: the 95% upper bound is ~0.105
    assert error_upper_bound(0, 0) == 1.0


def test_decided_answers_keep_their_error_on_new_data():
    table = RiskTable(fit(*synthetic(4000, 0)))
    probs, labels, qtypes = synthetic(20000, 1)
    for eps in (0.01, 0.02, 0.05, 0.1, 0.2):
        decided = [(p, y) for p, y in zip(probs, labels)
                   if table.apply({}, p, ["a", "b", "c"], "choice", eps)["decided"]]
        errors = sum(max(range(3), key=p.__getitem__) != y for p, y in decided)
        # the promise holds on data never seen when fitting; with too little evidence it decides nothing
        assert not decided or errors / len(decided) <= eps
        if eps >= 0.1:
            assert decided
    # a looser target decides more
    share = [sum(table.apply({}, p, ["a", "b", "c"], "choice", e)["decided"] for p in probs[:2000]) for e in (0.1, 0.2)]
    assert share[0] < share[1]


def test_sets_cover_the_truth_at_the_target_rate():
    table = RiskTable(fit(*synthetic(4000, 0)))
    probs, labels, _ = synthetic(20000, 2)
    for eps in (0.05, 0.1, 0.2):
        covered = sum(str(y) in table.apply({}, p, ["0", "1", "2"], "choice", eps)["set"] for p, y in zip(probs, labels))
        assert covered / len(probs) >= 1 - eps - 0.01


def test_thresholds_stop_at_the_first_failure_and_quantiles_are_conservative():
    assert fit_thresholds([0.9] * 10, [False] * 10, targets=(0.05,)) == {"0.05": None}  # never decide
    q = fit_set_quantiles([0.9, 0.8, 0.7], targets=(0.01,))
    assert q["0.01"] == 1.0  # too few points for 99%: the set is every option


def test_targets_below_the_table_promise_nothing():
    table = RiskTable(fit(*synthetic(500, 0)))
    a = table.apply({}, [0.99, 0.005, 0.005], ["a", "b", "c"], "choice", 0.001)
    assert a["decided"] is False and a["set"] == ["a", "b", "c"]


def test_api_reports_decided_and_sets(engine):
    table = RiskTable(fit(*synthetic(2000, 0)))
    req = {"state": "the customer asks about card delivery", "max_error": 0.05,
           "questions": {"intent": {"type": "choice", "instructions": "which intent",
                                    "criteria": {"card": None, "refund": None, "late": None}}}}
    with TestClient(create_app(Worker(engine), risk=table)) as c:
        r = c.post("/v1/systemone", json=req)
        assert r.status_code == 200, r.text
        a = r.json()["answers"]["intent"]
        assert isinstance(a["decided"], bool) and set(a["set"]) <= {"card", "refund", "late"} and a["set"]
        assert a["risk"]["max_error"] == 0.05 and a["risk"]["confidence_level"] == 0.95
        plain = c.post("/v1/systemone", json={k: v for k, v in req.items() if k != "max_error"}).json()
        assert "decided" not in plain["answers"]["intent"]  # opt-in: unchanged without max_error
        assert c.post("/v1/systemone", json={**req, "max_error": 0.7}).status_code == 422
    with TestClient(create_app(Worker(engine))) as c:
        assert c.post("/v1/systemone", json=req).status_code == 400  # no table on this server
