"""Convert more public datasets into Dragonfly training JSONL (A4: broader data, better transfer).

  python scripts/convert_public.py            # -> data/public-v1/<source>.jsonl (+ report.json)

Only permissive licences, so the trained checkpoints stay publishable:
  clinc150        clinc/clinc_oos (plus)                    CC-BY-3.0   intent: gold + 9 random intents + out-of-scope
  hellaswag       Rowan/hellaswag                           MIT         which ending continues the text
  snli            stanfordnlp/snli                          CC-BY-SA-4.0  entailment / neutral / contradiction
  civil-comments  google/civil_comments                     CC0-1.0     toxic? (clear cases only: >= 0.5 or 0.0)
  go-emotions     google-research-datasets/go_emotions      Apache-2.0  emotion, single-label comments only
Non-commercial sets (ANLI, Financial PhraseBank, ...) are deliberately left out.

Each source is capped (--cap, default 3000 records) and sampled with a fixed seed from its TRAIN split only, so no
source dominates the mix and the benchmark test splits are never touched. scripts/build_mix.py screens the result
against every evaluation set again.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data" / "public-v1"


def record(state, qid, question, source):
    return {"state": state, "questions": {qid: question}, "_meta": {"source": source}}


def sample(ds, n, seed, keep=lambda row: True):
    """First n rows (after a seeded shuffle of a streaming buffer) that pass keep()."""
    out = []
    for row in ds.shuffle(seed=seed, buffer_size=20_000):
        if keep(row):
            out.append(row)
            if len(out) >= n:
                break
    return out


def clinc150(cap, rng):
    ds = load_dataset("clinc/clinc_oos", "plus", split="train", streaming=True)
    names = ds.features["intent"].names
    oos = names.index("oos")
    rows = sample(ds, cap, 0)
    out = []
    for r in rows:
        gold = r["intent"]
        pool = [i for i in range(len(names)) if i not in (gold, oos)]
        options = rng.sample(pool, 9) + ([gold] if gold != oos else [rng.choice(pool)]) + [oos]
        rng.shuffle(options)
        criteria = {names[i].replace("_", " ") if i != oos else "out of scope": None for i in options}
        label = names[gold].replace("_", " ") if gold != oos else "out of scope"
        out.append(record(r["text"], "intent", {
            "type": "choice", "instructions": "Which intent does this banking/assistant request express?",
            "criteria": criteria, "label": label}, "clinc150"))
    return out


def hellaswag(cap, rng):
    rows = sample(load_dataset("Rowan/hellaswag", split="train", streaming=True), cap, 0,
                  lambda r: r["label"] not in ("", None))
    out = []
    for r in rows:
        endings = {f"{chr(65 + i)}: {e.strip()}": None for i, e in enumerate(r["endings"])}
        label = list(endings)[int(r["label"])]
        out.append(record(r["ctx"], "continuation", {
            "type": "choice", "instructions": "Which ending continues this text most plausibly?",
            "criteria": endings, "label": label}, "hellaswag"))
    return out


def snli(cap, rng):
    labels = ["entailment", "neutral", "contradiction"]
    rows = sample(load_dataset("stanfordnlp/snli", split="train", streaming=True), cap, 0, lambda r: r["label"] in (0, 1, 2))
    return [record(f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}", "relation", {
        "type": "choice", "instructions": "How does the hypothesis relate to the premise?",
        "criteria": {"entailment": "the premise implies it", "neutral": "cannot tell",
                     "contradiction": "the premise rules it out"},
        "label": labels[r["label"]]}, "snli") for r in rows]


def civil_comments(cap, rng):
    ds = load_dataset("google/civil_comments", split="train", streaming=True)
    toxic = sample(ds, cap // 2, 0, lambda r: r["toxicity"] >= 0.5 and 20 < len(r["text"]) < 1500)
    clean = sample(ds, cap - len(toxic), 1, lambda r: r["toxicity"] == 0.0 and 20 < len(r["text"]) < 1500)
    return [record(r["text"], "toxic", {"type": "noul", "instructions": "Is this comment toxic (rude, disrespectful "
                   "or hateful)?", "label": r["toxicity"] >= 0.5}, "civil-comments") for r in toxic + clean]


def go_emotions(cap, rng):
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="train", streaming=True)
    names = ds.features["labels"].feature.names
    rows = sample(ds, cap, 0, lambda r: len(r["labels"]) == 1 and names[r["labels"][0]] != "neutral")
    out = []
    for r in rows:
        gold = names[r["labels"][0]]
        pool = [n for n in names if n not in (gold, "neutral")]
        options = rng.sample(pool, 6) + [gold]
        rng.shuffle(options)
        out.append(record(r["text"], "emotion", {
            "type": "choice", "instructions": "Which emotion does the writer express?",
            "criteria": {o: None for o in options}, "label": gold}, "go-emotions"))
    return out


SOURCES = {"clinc150": clinc150, "hellaswag": hellaswag, "snli": snli, "civil-comments": civil_comments,
           "go-emotions": go_emotions}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--only", nargs="*", default=list(SOURCES))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in a.only:
        rows = SOURCES[name](a.cap, random.Random(0))
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        labels = Counter(str(next(iter(r["questions"].values()))["label"]) for r in rows)
        report[name] = {"records": len(rows), "distinct_labels": len(labels), "top_labels": labels.most_common(3)}
        print(name, report[name])
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
