"""Labelled training and evaluation data.

One JSON object per line: a /v1/systemone request plus a label per question, the same format as Kev's eval suites
(github.com/jaredpalmer/kev, evals/*), so results compare directly:

  {"state": "...", "questions": {"intent": {"type": "choice", "instructions": "...", "criteria": {...},
                                            "label": "card_arrival", "target": {"card_arrival": 0.9, ...}}}}

label: the criteria name (choice), true/false (noul), or the level index (score).
target (optional): a soft distribution keyed like the probabilities; used instead of the label for training.
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import DecideRequest, to_record


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if "state" not in r or not isinstance(r.get("questions"), dict) or not r["questions"]:
                raise ValueError(f"{path}:{n}: a record needs a state and a non-empty questions object")
            for qid, q in r["questions"].items():
                if "label" not in q:
                    raise ValueError(f"{path}:{n}: question {qid!r} has no label")
            records.append(r)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def materialize(labelled: dict) -> dict:
    """Labelled record -> internal record (the serving path's to_record) with int labels and optional soft targets."""
    request = DecideRequest.model_validate({
        "state": labelled["state"],
        "questions": {qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
                      for qid, q in labelled["questions"].items()},
    })
    record, meta = to_record(request)
    for q, m, src in zip(record["questions"], meta, labelled["questions"].values()):
        y = src["label"]
        q["label"] = m["keys"].index(y) if m["type"] == "choice" else int(y)
        q["source"] = labelled.get("_meta", {}).get("source", "custom")
        if src.get("target") is not None:
            t = [float(src["target"].get(k, 0.0)) for k in m["keys"]]
            if sum(t) <= 0:
                raise ValueError(f"target for {m['id']} puts no mass on any option")
            q["target"] = [x / sum(t) for x in t]
    return record


def load_records(path: str | Path) -> list[dict]:
    return [materialize(r) for r in load_jsonl(path)]
