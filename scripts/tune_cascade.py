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
    probs = [torch.softmax(z / engine.config.temperature, -1).tolist() for z in logits]
    conf = [question_confidence(q["qtype"], p) for p, q in zip(probs, rows)]
    right = [max(range(len(p)), key=p.__getitem__) == q["label"] for p, q in zip(probs, rows)]
    return conf, right


def cascade(conf_s, right_s, right_m, t):
    escalate = [c < t for c in conf_s]
    acc = sum(rm if e else rs for e, rs, rm in zip(escalate, right_s, right_m)) / len(right_s)
    return acc, sum(escalate) / len(escalate)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--tolerance", type=float, default=0.01, help="accuracy allowed below tier M alone")
    a = ap.parse_args()

    s, m = Engine.load(a.small), Engine.load(a.large)
    splits = {}
    for name, path in (("calibration", a.calibration), ("test", a.test)):
        recs = load_records(path)
        conf_s, right_s = per_question(s, recs)
        _, right_m = per_question(m, recs)
        splits[name] = (conf_s, right_s, right_m)

    conf_s, right_s, right_m = splits["calibration"]
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

    conf_s, right_s, right_m = splits["test"]
    acc, esc = cascade(conf_s, right_s, right_m, chosen)
    result = {"threshold": chosen, "tolerance": a.tolerance,
              "test": {"cascade_accuracy": round(acc, 4), "escalated": round(esc, 4),
                       "s_alone": round(sum(right_s) / len(right_s), 4), "m_alone": round(sum(right_m) / len(right_m), 4)},
              "calibration_table": table}
    print(f"\nchosen threshold {chosen:.2f} (on calibration) -> test: cascade {acc:.4f} with {esc:.1%} escalated "
          f"(S alone {result['test']['s_alone']:.4f}, M alone {result['test']['m_alone']:.4f})")
    json.dump(result, open("runs/cascade.json", "w"), indent=2)


if __name__ == "__main__":
    main()
