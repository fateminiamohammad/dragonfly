"""Head-to-head: any number of /v1/systemone servers (Dragonfly, Kev, Jev) on the same labelled requests.

All of them speak the TypeSafe System One API, so each gets byte-identical request bodies, sent the same way, and is
scored against the same labels. Latency is client-side round trip (network included, which matters for hosted Jev).

  python bench/compare_systemone.py --data data/decision-v2/test.jsonl --n 200 \\
      --system dragonfly=http://localhost:8000@DRAGONFLY_API_KEY \\
      --system kev-4b=http://localhost:8009 \\
      --system jev=https://api.typesafe.ai@TYPESAFE_API_KEY

A system is name=base_url, optionally @ENV_VAR naming the environment variable that holds its bearer key. A system
whose key variable is unset is skipped with a note (so Jev is included only when you have a TypeSafe key). Dragonfly
requests carry Cache-Control: no-cache so its answer cache can't flatter it; other servers ignore the header.
Kev and Jev answers are read in either shape the API allows (noul as "noul" or "probability").
"""

import argparse
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def api_body(record: dict, model: str) -> dict:
    return {"state": record["state"], "model": model, "questions": {
        qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")} for qid, q in record["questions"].items()}}


def gold(q: dict) -> str:
    if q["type"] == "noul":
        return "true" if q["label"] in (True, "true", 1) else "false"
    return str(q["label"])


def predicted(answer: dict) -> str:
    kind = answer.get("type")
    if kind == "noul" or "noul" in answer or ("probability" in answer and "probabilities" not in answer):
        p = answer.get("noul", answer.get("probability"))
        return "true" if p >= 0.5 else "false"
    if kind == "choice" and "choice" in answer:
        return str(answer["choice"])
    probs = answer["probabilities"]
    return max(probs, key=probs.get)


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


class System:
    def __init__(self, spec: str):
        name, rest = spec.split("=", 1)
        url, _, key_env = rest.partition("@")
        self.name, self.url, self.key_env = name, url.rstrip("/"), key_env
        self.key = os.environ.get(key_env) if key_env else None
        self.model = "jev-latest" if "typesafe.ai" in url else "dragonfly-latest" if "dragonfly" in name else "kev-latest"

    @property
    def available(self) -> bool:
        return not self.key_env or bool(self.key)

    def ask(self, record: dict) -> tuple[dict, float, float | None]:
        headers = {"content-type": "application/json", "cache-control": "no-cache"}
        if self.key:
            headers["authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(f"{self.url}/v1/systemone", data=json.dumps(api_body(record, self.model)).encode(),
                                     headers=headers, method="POST")
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        return out, (time.perf_counter() - t) * 1000, out.get("latency_ms")


def run(system: System, records: list[dict], concurrency: int, warm: list[dict]) -> dict:
    for r in warm:  # not measured: model load, graph capture, connection setup
        system.ask(r)

    def one(r):
        try:
            out, ms, server_ms = system.ask(r)
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            return None, str(e)
        right = sum(predicted(out["answers"][qid]) == gold(q) for qid, q in r["questions"].items())
        return (ms, server_ms, right, len(r["questions"])), None

    started = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        results = list(pool.map(one, records))
    wall = time.perf_counter() - started
    ok = [x for x, err in results if x]
    errors = [err for x, err in results if err]
    lat = [x[0] for x in ok]
    server = [x[1] for x in ok if x[1] is not None]
    right, total = sum(x[2] for x in ok), sum(x[3] for x in ok)
    return {"system": system.name, "requests": len(ok), "errors": len(errors), "first_error": errors[0] if errors else None,
            "questions": total, "accuracy": round(right / max(total, 1), 4),
            "p50_ms": round(pct(lat, 50), 1) if lat else None, "p95_ms": round(pct(lat, 95), 1) if lat else None,
            "mean_ms": round(statistics.mean(lat), 1) if lat else None,
            "server_p50_ms": round(pct(server, 50), 1) if server else None,
            "throughput_rps": round(len(ok) / wall, 1), "concurrency": concurrency}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--system", action="append", required=True, help="name=base_url[@ENV_VAR_WITH_KEY]")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default="runs/compare-systemone.json")
    a = ap.parse_args()

    records = [json.loads(line) for line in open(a.data, encoding="utf-8") if line.strip()]
    records = random.Random(a.seed).sample(records, min(a.n + a.warmup, len(records)))
    warm, test = records[:a.warmup], records[a.warmup:]
    systems = [System(s) for s in a.system]
    results = []
    for s in systems:
        if not s.available:
            print(f"skipping {s.name}: set {s.key_env} to include it")
            continue
        print(f"running {s.name} ({s.url}) on {len(test)} requests, concurrency {a.concurrency}...", flush=True)
        results.append(run(s, test, a.concurrency, warm) | {"requests_sent": len(test)})

    base = next((r for r in results if r["system"].startswith("dragonfly")), results[0] if results else None)
    print(f"\n{len(test)} requests ({sum(len(r['questions']) for r in test)} questions) from {a.data}, "
          f"concurrency {a.concurrency}")
    print(f"{'system':14}{'accuracy':>9}{'p50 ms':>9}{'p95 ms':>9}{'req/s':>8}{'errors':>7}  {'Dragonfly vs it':>16}")
    for r in results:
        ratio = "-" if r is base else f"{r['p50_ms'] / base['p50_ms']:.1f}x faster" if r["p50_ms"] and base["p50_ms"] else "n/a"
        print(f"{r['system']:14}{r['accuracy']:9.3f}{r['p50_ms'] or 0:9.1f}{r['p95_ms'] or 0:9.1f}"
              f"{r['throughput_rps']:8.1f}{r['errors']:7d}  {ratio:>16}")
        if r["first_error"]:
            print(f"   first error: {r['first_error'][:120]}")
    json.dump({"data": a.data, "n": len(test), "seed": a.seed, "concurrency": a.concurrency, "results": results},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
