"""Download the typed-decisions benchmark (LocalLLaMA/typed-decisions, Apache-2.0) and convert it to Dragonfly's labelled
JSONL (a /v1/systemone request plus a label per question, and the gold distribution as a soft target).

  python scripts/convert_typed_decisions.py            # -> data/typed-decisions/{train,test}.jsonl

Its public leaderboard scores TypeSafe Jev 1.13.0 on the same test split (400 cases, 2,000 decisions), so Dragonfly,
Kev and Laya can be compared against Jev's published accuracy without a TypeSafe key.
"""

import json
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/main/all"
OUT = Path(__file__).resolve().parent.parent / "data" / "typed-decisions"


def load(value):
    return json.loads(value) if isinstance(value, str) else value


def convert(row: dict) -> dict:
    questions, gold = load(row["questions"]), load(row["gold"])
    for qid, q in questions.items():
        g = gold[qid]
        if q["type"] == "noul":
            q["label"] = g["label"] == "true"
        elif q["type"] == "score":
            q["label"] = int(g["label"])
        else:
            q["label"] = g["label"]
        q["target"] = g["probabilities"]
    return {"state": load(row["state"]), "questions": questions,
            "_meta": {"source": row["workflow"], "id": row["id"], "split": row["split"]}}


def main():
    import pyarrow.parquet as pq

    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        parquet = OUT / f"{split}.parquet"
        if not parquet.exists():
            urllib.request.urlretrieve(f"{BASE}/{split}-00000-of-00001.parquet", parquet)
        rows = pq.read_table(parquet).to_pylist()
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(convert(row), ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} cases, {sum(r['n_questions'] for r in rows)} decisions -> {OUT / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
