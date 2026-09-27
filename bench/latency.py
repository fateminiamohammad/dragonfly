"""Latency benchmark for a running Dragonfly (or any /v1/systemone-compatible) server.

  python bench/latency.py --url http://localhost:8000 --requests 500 --concurrency 1
  python bench/latency.py --url http://localhost:8000 --requests 2000 --concurrency 32

Reports end-to-end client latency (HTTP included) and the server's own model time (latency_ms in the response).
Each request gets a unique state by default so the answer cache cannot serve it; --allow-cache measures cache hits.
Comparison against LLM APIs arrives with the bench service (M3).
"""

import argparse
import json
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_REQUEST = Path(__file__).resolve().parent.parent / "model-service" / "examples" / "request.json"


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--request", default=str(DEFAULT_REQUEST))
    ap.add_argument("--requests", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--api-key")
    ap.add_argument("--allow-cache", action="store_true", help="send identical requests (measures cache hits)")
    a = ap.parse_args()

    template = json.loads(Path(a.request).read_bytes())
    headers = {"content-type": "application/json"}
    if a.api_key:
        headers["authorization"] = f"Bearer {a.api_key}"
    n_questions = len(template["questions"])

    run = uuid.uuid4().hex[:8]  # unique per run, so a second run cannot hit the first run's cached answers

    def body_for(i: int) -> bytes:
        if a.allow_cache:
            return json.dumps(template).encode()
        state = template["state"]
        unique = {**state, "request_id": f"{run}-{i}"} if isinstance(state, dict) else f"{state} (request {run}-{i})"
        return json.dumps({**template, "state": unique}).encode()

    def one(i):
        req = urllib.request.Request(f"{a.url}/v1/systemone", data=body_for(i), headers=headers, method="POST")
        t = time.perf_counter()
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        return (time.perf_counter() - t) * 1000, out["latency_ms"]

    for i in range(a.warmup):
        one(-1 - i)
    started = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as pool:
        results = list(pool.map(one, range(a.requests)))
    wall = time.perf_counter() - started

    client = [r[0] for r in results]
    model = [r[1] for r in results]
    mode = "identical requests (cache hits)" if a.allow_cache else "unique requests (no cache)"
    print(f"{a.requests} requests x {n_questions} questions, concurrency {a.concurrency}, {mode}")
    print(f"  client latency ms   p50 {pct(client, 50):7.2f}   p95 {pct(client, 95):7.2f}   mean {statistics.mean(client):7.2f}")
    print(f"  model time ms       p50 {pct(model, 50):7.2f}   p95 {pct(model, 95):7.2f}")
    print(f"  throughput          {a.requests / wall:7.1f} req/s   {a.requests * n_questions / wall:7.1f} questions/s")


if __name__ == "__main__":
    main()
