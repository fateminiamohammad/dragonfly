"""Fit the risk table behind "max_error" (dragonfly/risk.py) on the calibration split, then check it once on test.

  python scripts/fit_risk.py --small runs/dragonfly-s2 --large runs/dragonfly-m4b --threshold 0.70 \
      --calibration data/mix-v1/calibration.jsonl --test data/decision-v2/test.jsonl --out runs/risk.json

The probabilities are the served ones: the S -> M cascade at the served threshold (or one checkpoint with --small
only). For every target error the report shows, on the test split: the realized error of "decided" answers (it
should stay at or below the target), the share of questions decided, and the coverage and mean size of the sets.
Serve it with DRAGONFLY_RISK=<out>.
"""

import argparse
import json

from dragonfly.data import load_records
from dragonfly.engine import Cascade, Engine
from dragonfly.risk import RiskTable, fit


def served_probs(engine, records, batch=16):
    probs, labels, qtypes = [], [], []
    for i in range(0, len(records), batch):
        chunk = records[i:i + batch]
        p, _, _ = engine.probs(chunk)
        for rec, rp in zip(chunk, p):
            for q, qp in zip(rec["questions"], rp):
                probs.append(qp)
                labels.append(q["label"])
                qtypes.append(q["qtype"])
    return probs, labels, qtypes


def report(table: RiskTable, probs, labels, qtypes) -> dict:
    out = {}
    for eps in table.targets:
        decided = wrong = covered = size = 0
        for p, y, qt in zip(probs, labels, qtypes):
            keys = [str(i) for i in range(len(p))]
            a = table.apply({}, p, keys, qt, eps)
            top = max(range(len(p)), key=p.__getitem__)
            if a["decided"]:
                decided += 1
                wrong += top != y
            covered += str(y) in a["set"]
            size += len(a["set"])
        n = len(probs)
        out[str(eps)] = {"decided_share": round(decided / n, 4),
                         "error_when_decided": round(wrong / decided, 4) if decided else None,
                         "set_coverage": round(covered / n, 4), "mean_set_size": round(size / n, 2)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", required=True)
    ap.add_argument("--large")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="runs/risk.json")
    a = ap.parse_args()

    engine = Engine.load(a.small)
    if a.large:
        engine = Cascade(engine, Engine.load(a.large), a.threshold)
    data = fit(*served_probs(engine, load_records(a.calibration)))
    data["fitted_on"] = {"small": a.small, "large": a.large, "threshold": a.threshold if a.large else None,
                         "calibration": a.calibration}
    table = RiskTable(data)
    data["test_report"] = {"data": a.test, **report(table, *served_probs(engine, load_records(a.test)))}
    with open(a.out, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data["test_report"], indent=2))


if __name__ == "__main__":
    main()
