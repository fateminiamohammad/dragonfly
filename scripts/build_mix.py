"""Build a larger, deduplicated training mix for Dragonfly and screen it against every evaluation set.

  python scripts/build_mix.py        # -> data/mix-v1/{train,calibration}.jsonl + data/mix-v1/report.json

Sources (all public; each record keeps its source dataset's licence, see the Kev suite manifests):
  decision-v2 train                 data/decision-v2/train.jsonl          (scripts/fetch_data.py)
  Kev decision-v8 train             data/kev-suites/v8/decision-v8/        (hash-pinned Hub mirror, verified)
  Kev public-pool-v6 train          data/kev-suites/public-pool-v6/        (adds ARC, OpenBookQA, CommonsenseQA)
  Kev devtools-v1 train             data/kev-suites/devtools-v1/           (code review, commits, flaky tests)
  Kev longstate-v2 train            data/kev-suites/longstate-v2/          (long documents)
  typed-decisions train             data/typed-decisions/train.jsonl       (scripts/convert_typed_decisions.py)

Screening: a training record is dropped when its normalised state (or Kev's text hash) matches any record in the
decision-v2 test/calibration sets or the typed-decisions test set (strict: the document itself must be unseen). Among
training records, exact duplicates (same document AND same questions) are dropped; the same document asked different
questions is kept, because it is new supervision.
typed-decisions has no calibration split: 100 of its train cases are held out (seeded) as calibration.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
SOURCES = [
    ("decision-v2", ROOT / "decision-v2/train.jsonl"),
    ("kev-decision-v8", ROOT / "kev-suites/v8/decision-v8/train.jsonl"),
    ("kev-public-pool-v6", ROOT / "kev-suites/public-pool-v6/train.jsonl"),
    ("kev-devtools-v1", ROOT / "kev-suites/devtools-v1/train.jsonl"),
    ("kev-longstate-v2", ROOT / "kev-suites/longstate-v2/train.jsonl"),
    ("typed-decisions", ROOT / "typed-decisions/train.jsonl"),
]
SCREEN = [ROOT / "decision-v2/test.jsonl", ROOT / "decision-v2/calibration.jsonl", ROOT / "typed-decisions/test.jsonl"]
TD_CALIBRATION_CASES = 100


def norm_state(state) -> str:
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def keys(record: dict) -> set[str]:
    k = {norm_state(record["state"])}
    if record.get("_meta", {}).get("text_sha256"):
        k.add(record["_meta"]["text_sha256"])
    return k


def question_sig(record: dict) -> str:
    """The questions a record asks (not their labels): the same document with different questions is new data."""
    qs = [{k: q.get(k) for k in ("type", "instructions", "criteria")} for q in record["questions"].values()]
    return hashlib.sha256(json.dumps(qs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path: Path) -> list[dict]:
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "state" in r and isinstance(r.get("questions"), dict) and r["questions"] and \
                all("label" in q for q in r["questions"].values()):
            out.append(r)
    return out


def main():
    screened: set[str] = set()
    for path in SCREEN:
        for r in read(path):
            screened |= keys(r)

    out = ROOT / "mix-v1"
    out.mkdir(parents=True, exist_ok=True)
    train, calibration, report = [], [], {}
    seen: set[str] = set()
    for name, path in SOURCES:
        records = read(path)
        if name == "typed-decisions":
            rng = random.Random(20260928)
            rng.shuffle(records)
            calibration += records[:TD_CALIBRATION_CASES]
            screened |= set().union(*(keys(r) for r in records[:TD_CALIBRATION_CASES]))
            records = records[TD_CALIBRATION_CASES:]
        kept = dropped_test = dropped_dup = 0
        types = Counter()
        for r in records:
            k = keys(r)
            if k & screened:
                dropped_test += 1
                continue
            dup = {x + question_sig(r) for x in k}
            if dup & seen:
                dropped_dup += 1
                continue
            seen |= dup
            r.setdefault("_meta", {})["mix_source"] = name
            if name != "decision-v2" and "source" not in r["_meta"]:
                r["_meta"]["source"] = name
            train.append(r)
            kept += 1
            types.update(q["type"] for q in r["questions"].values())
        report[name] = {"records_in": len(records), "kept": kept, "questions": sum(types.values()),
                        "dropped_overlaps_eval": dropped_test, "dropped_duplicates": dropped_dup, "types": dict(types)}

    calibration = read(ROOT / "decision-v2/calibration.jsonl") + calibration
    random.Random(7).shuffle(train)
    for name, rows in (("train", train), ("calibration", calibration)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["total"] = {"train_records": len(train), "train_questions": sum(len(r["questions"]) for r in train),
                       "calibration_records": len(calibration),
                       "calibration_questions": sum(len(r["questions"]) for r in calibration)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    for name, s in report.items():
        print(f"{name:20} {json.dumps(s)}")


if __name__ == "__main__":
    main()
