"""Write a training set whose targets are a teacher checkpoint's calibrated probabilities (knowledge distillation).

  python scripts/distill_targets.py --teacher runs/dragonfly-m --data data/decision-v2/train.jsonl \
      --out data/decision-v2/train.distilled.jsonl --mix 0.5

Each question keeps its label and gains a soft `target`: mix * one-hot(label) + (1 - mix) * teacher probabilities.
dragonfly-train uses `target` when present, so training tier S on this file transfers tier M's knowledge and its
calibrated uncertainty into the fast tier. The label mix keeps the student anchored to ground truth where the teacher
is wrong.
"""

import argparse
import json

import torch

from dragonfly.data import load_jsonl, materialize
from dragonfly.engine import Engine
from dragonfly.schema import question_keys
from dragonfly.training.evaluate import collect_logits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix", type=float, default=0.5, help="weight of the one-hot label in the target")
    a = ap.parse_args()

    teacher = Engine.load(a.teacher)
    raw = load_jsonl(a.data)
    rows, states = [], []
    for r in raw:
        rec = materialize(r)
        for q in rec["questions"]:
            rows.append(q)
            states.append(rec["state"])
    logits = collect_logits(teacher, rows, states)
    probs = [torch.softmax(z / teacher.config.temperature, -1).tolist() for z in logits]

    n = 0
    for r in raw:  # same order as the rows above: records, then their questions
        for q in r["questions"].values():
            keys = question_keys(q["type"], q.get("criteria"))
            p = probs[n]
            label = keys.index(q["label"]) if q["type"] == "choice" else int(q["label"])
            q["target"] = {k: round(a.mix * (j == label) + (1 - a.mix) * p[j], 6) for j, k in enumerate(keys)}
            n += 1
    assert n == len(probs), (n, len(probs))
    with open(a.out, "w", encoding="utf-8") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    agree = sum(max(range(len(p)), key=p.__getitem__) == q["label"] for p, q in zip(probs, rows)) / len(rows)
    print(f"wrote {a.out}: {len(raw)} records, {n} questions; teacher agrees with labels on {agree:.1%}")


if __name__ == "__main__":
    main()
