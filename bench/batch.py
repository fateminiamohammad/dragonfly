"""Batch throughput: decisions per second through POST /v1/batch vs the same requests one by one.

  python bench/batch.py --url http://127.0.0.1:8000 --api-key <key> --data data/decision-v2/test.jsonl --n 1000

Labels are stripped; every request carries Cache-Control: no-cache, so the answer cache cannot serve any of them.
"""

import argparse
import json
import time
import urllib.request


def post(url, body, key, timeout=3600):
    headers = {"content-type": "application/json", "cache-control": "no-cache"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def strip(record):
    qs = {qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")}
          for qid, q in record["questions"].items()}
    return {"state": record["state"], "questions": qs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key")
    ap.add_argument("--data", default="data/decision-v2/test.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--sequential", type=int, default=200, help="requests to time one by one for comparison")
    a = ap.parse_args()
    requests = [strip(json.loads(line)) for line in open(a.data, encoding="utf-8") if line.strip()][: a.n]
    questions = sum(len(r["questions"]) for r in requests)

    post(f"{a.url}/v1/batch", {"requests": requests[:20]}, a.api_key)  # warm-up
    t = time.perf_counter()
    out = post(f"{a.url}/v1/batch", {"requests": requests}, a.api_key)
    batch_s = time.perf_counter() - t
    errors = sum("error" in r for r in out["results"])

    seq = requests[: a.sequential]
    t = time.perf_counter()
    for r in seq:
        post(f"{a.url}/v1/systemone", r, a.api_key)
    seq_s = time.perf_counter() - t
    seq_q = sum(len(r["questions"]) for r in seq)

    result = {
        "requests": len(requests), "questions": questions, "errors": errors,
        "batch_seconds": round(batch_s, 2), "batch_decisions_per_s": round(questions / batch_s, 1),
        "one_by_one_decisions_per_s": round(seq_q / seq_s, 1),
        "speedup": round((questions / batch_s) / (seq_q / seq_s), 1),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
