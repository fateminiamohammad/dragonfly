"""Pick the S -> M cascade threshold on held-out data, then report it once on the test split.

  python scripts/tune_cascade.py --small runs/dragonfly-s --large runs/dragonfly-m \
      --calibration data/decision-v2/calibration.jsonl --test data/decision-v2/test.jsonl

For each threshold t, questions whose tier-S confidence is below t go to tier M. The script prints accuracy and the
escalation rate (the share of questions that pay tier M's latency) per threshold on the calibration split, picks the
smallest threshold that gets within --tolerance of tier M's accuracy, and evaluates that single choice on the test split.
"""

import argparse
import json

import torch

from dragonfly.data import load_records
from dragonfly.engine import Engine
from dragonfly.schema import question_confidence
from dragonfly.training.evaluate import collect_logits, question_rows


def per_question(engine: Engine, records: list[dict]):
    rows, states = question_rows(records)
    logits = collect_logits(engine, rows, states)
    probs = [torch.softmax(z / engine.temperature_for(q["qtype"]), -1).tolist() for z, q in zip(logits, rows)]
    conf = [question_confidence(q["qtype"], p) for p, q in zip(probs, rows)]
    right = [max(range(len(p)), key=p.__getitem__) == q["label"] for p, q in zip(probs, rows)]
    return conf, right, [q["qtype"] for q in rows]


def cascade(conf_s, right_s, right_m, t, qtypes=None):
    """t: one threshold, or one per question type."""
    escalate = [c < (t[q] if isinstance(t, dict) else t) for c, q in zip(conf_s, qtypes or [None] * len(conf_s))]
    acc = sum(rm if e else rs for e, rs, rm in zip(escalate, right_s, right_m)) / len(right_s)
    return acc, sum(escalate) / len(escalate)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--tolerance", type=float, default=0.01, help="accuracy allowed below tier M alone")
    ap.add_argument("--out", default="runs/cascade.json", help="where to write the result")
    ap.add_argument("--per-type", action="store_true",
                    help="also pick one threshold per question type (serve as noul=..,choice=..,score=..)")
    a = ap.parse_args()

    s, m = Engine.load(a.small), Engine.load(a.large)
    splits = {}
    for name, path in (("calibration", a.calibration), ("test", a.test)):
        recs = load_records(path)
        conf_s, right_s, qtypes = per_question(s, recs)
        _, right_m, _ = per_question(m, recs)
        splits[name] = (conf_s, right_s, right_m, qtypes)

    conf_s, right_s, right_m, qtypes = splits["calibration"]
    acc_m = sum(right_m) / len(right_m)
    print(f"calibration split: S alone {sum(right_s) / len(right_s):.4f}, M alone {acc_m:.4f}")
    print(f"{'threshold':>9} {'accuracy':>9} {'escalated':>10}")
    table, chosen = [], None
    for t in [x / 20 for x in range(0, 21)]:
        acc, esc = cascade(conf_s, right_s, right_m, t)
        table.append({"threshold": t, "accuracy": round(acc, 4), "escalated": round(esc, 4)})
        print(f"{t:9.2f} {acc:9.4f} {esc:10.1%}")
        if chosen is None and acc >= acc_m - a.tolerance:
            chosen = t
    chosen = 1.0 if chosen is None else chosen

    per_type = None
    if a.per_type:
        per_type = {}
        for qt in sorted(set(qtypes)):
            idx = [i for i, q in enumerate(qtypes) if q == qt]
            sub = [[v[i] for i in idx] for v in (conf_s, right_s, right_m)]
            acc_m_t = sum(sub[2]) / len(idx)
            per_type[qt] = next((t for t in [x / 20 for x in range(21)]
                                 if cascade(*sub, t)[0] >= acc_m_t - a.tolerance), 1.0)
        print("per-type thresholds (calibration):", per_type)

    conf_s, right_s, right_m, qtypes = splits["test"]
    acc, esc = cascade(conf_s, right_s, right_m, chosen)
    result = {"threshold": chosen, "tolerance": a.tolerance,
              "test": {"cascade_accuracy": round(acc, 4), "escalated": round(esc, 4),
                       "s_alone": round(sum(right_s) / len(right_s), 4), "m_alone": round(sum(right_m) / len(right_m), 4)},
              "calibration_table": table}
    if per_type:
        acc_t, esc_t = cascade(conf_s, right_s, right_m, per_type, qtypes)
        result["per_type"] = {"thresholds": per_type, "serve_as": ",".join(f"{k}={v}" for k, v in per_type.items()),
                              "test": {"cascade_accuracy": round(acc_t, 4), "escalated": round(esc_t, 4)}}
        print(f"per-type -> test: cascade {acc_t:.4f} with {esc_t:.1%} escalated")
    print(f"\nchosen threshold {chosen:.2f} (on calibration) -> test: cascade {acc:.4f} with {esc:.1%} escalated "
          f"(S alone {result['test']['s_alone']:.4f}, M alone {result['test']['m_alone']:.4f})")
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
