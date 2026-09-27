"""Evaluate a checkpoint on labelled data: accuracy, calibration (ECE, Brier, NLL), share automatable at a 5% error
budget, and option-order sensitivity. `dragonfly-eval --checkpoint runs/s --data test.jsonl`"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import torch

from ..calibration import auto_rate, brier, ece, nll
from ..data import load_records
from ..engine import Engine
from ..schema import question_confidence


def question_rows(records: list[dict]) -> tuple[list[dict], list[str]]:
    rows, states = [], []
    for rec in records:
        for q in rec["questions"]:
            rows.append(q)
            states.append(rec["state"])
    return rows, states


@torch.inference_mode()
def collect_logits(engine: Engine, rows: list[dict], states: list[str], batch_size: int = 32) -> list[torch.Tensor]:
    """Raw (untempered) logits per question row, trimmed to its option count."""
    out = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch, _ = engine.encode([{"state": s, "questions": [q]} for q, s in zip(chunk, states[start:start + batch_size])])
        logits = engine.logits(batch).float().cpu()
        out.extend(row[: len(q["options"])] for row, q in zip(logits, chunk))
    return out


def permuted(rows: list[dict], seed: int = 0) -> tuple[list[dict], list[list[int]]]:
    """Choice rows with their options shuffled, and each row's permutation (new position -> old index)."""
    rng = random.Random(seed)
    out, perms = [], []
    for q in rows:
        perm = list(range(len(q["options"])))
        rng.shuffle(perm)
        out.append({**q, "options": [q["options"][i] for i in perm]})
        perms.append(perm)
    return out, perms


def summarize(probs: list[list[float]], rows: list[dict]) -> dict:
    labels = [q["label"] for q in rows]
    conf = [question_confidence(q["qtype"], p) for p, q in zip(probs, rows)]
    top = [max(p) for p in probs]  # calibration of the argmax probability (standard top-label ECE)
    correct = [max(range(len(p)), key=p.__getitem__) == y for p, y in zip(probs, labels)]
    return {
        "n": len(rows),
        "accuracy": round(sum(correct) / max(len(rows), 1), 4),
        "ece": round(ece(top, correct), 4),
        "brier": round(brier(probs, labels), 4),
        "nll": round(nll(probs, labels), 4),
        "auto_rate@5%": round(auto_rate(conf, correct, 0.05), 4),
    }


def evaluate(engine: Engine, records: list[dict], order_check: bool = True) -> dict:
    rows, states = question_rows(records)
    logits = collect_logits(engine, rows, states)
    t = engine.config.temperature
    probs = [torch.softmax(z / t, -1).tolist() for z in logits]
    result = {"overall": summarize(probs, rows), "by_source": {}, "by_type": {}}
    for key, field in (("by_source", "source"), ("by_type", "qtype")):
        groups = defaultdict(list)
        for i, q in enumerate(rows):
            groups[q.get(field, "custom")].append(i)
        result[key] = {g: summarize([probs[i] for i in idx], [rows[i] for i in idx]) for g, idx in sorted(groups.items())}

    if order_check:
        idx = [i for i, q in enumerate(rows) if q["qtype"] == "choice" and len(q["options"]) > 1]
        if idx:
            shuffled, perms = permuted([rows[i] for i in idx])
            z2 = collect_logits(engine, shuffled, [states[i] for i in idx])
            flips = 0
            for j, i in enumerate(idx):
                new_best = perms[j][int(z2[j].argmax())]
                flips += new_best != int(logits[i].argmax())
            result["order_flip_rate"] = round(flips / len(idx), 4)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="labelled JSONL")
    ap.add_argument("--device")
    ap.add_argument("--no-order-check", action="store_true")
    a = ap.parse_args()
    engine = Engine.load(a.checkpoint, a.device)
    print(json.dumps(evaluate(engine, load_records(a.data), not a.no_order_check), indent=2))


if __name__ == "__main__":
    main()
