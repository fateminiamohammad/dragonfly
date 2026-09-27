"""Dragonfly vs an LLM on the same labelled requests: latency and accuracy, measured the same way.

Two LLM modes, because both numbers matter and publishing only one would be misleading:
  reason       the LLM may think, then outputs JSON answers (how LLMs are often used; where "200x" comes from)
  constrained  the LLM outputs only the JSON answers, max 64 tokens, temperature 0 (the fair comparison)

Works with any OpenAI-compatible chat API (OpenAI, Anthropic's OpenAI-compatible endpoint, vLLM, Ollama, ...):

  export LLM_BASE_URL=https://api.openai.com/v1 LLM_API_KEY=... LLM_MODEL=...
  python bench/compare_llm.py --data data/decision-v2/test.jsonl --n 50 --dragonfly http://localhost:8000

Every request is sent sequentially (one at a time) to both systems, so latencies are comparable.
"""

import argparse
import json
import os
import random
import re
import statistics
import time
import urllib.request

SYSTEM = {
    "reason": "You answer typed questions about a document. Think step by step first. Then, on the last line, output "
              "only a JSON object mapping each question id to its answer key.",
    "constrained": "You answer typed questions about a document. Output only a JSON object mapping each question id to "
                   "its answer key. No other text.",
}


def answer_keys(q: dict) -> list[str]:
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(q["criteria"]))]


def gold(q: dict) -> str:
    if q["type"] == "noul":
        return "true" if q["label"] in (True, "true", 1) else "false"
    return str(q["label"])


def prompt(record: dict) -> str:
    state = record["state"] if isinstance(record["state"], str) else json.dumps(record["state"], ensure_ascii=False)
    lines = [f"Document:\n{state}", ""]
    for qid, q in record["questions"].items():
        lines.append(f"Question id: {qid}")
        if q.get("instructions"):
            lines.append(f"Question: {q['instructions']}")
        if q["type"] == "choice":
            opts = [k if v in (None, "") else f"{k}: {v}" for k, v in q["criteria"].items()]
            lines.append("Answer with one key: " + "; ".join(opts))
        elif q["type"] == "noul":
            lines.append('Answer with "true" or "false".')
        else:
            lines.append("Answer with the level index: " + "; ".join(f"{i} = {c}" for i, c in enumerate(q["criteria"])))
        lines.append("")
    return "\n".join(lines)


def post(url: str, body: dict, headers: dict, timeout: float = 600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **headers})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t) * 1000


def ask_llm(record: dict, mode: str, a) -> tuple[dict, float]:
    body = {"model": a.llm_model, "temperature": 0,
            "messages": [{"role": "system", "content": SYSTEM[mode]}, {"role": "user", "content": prompt(record)}]}
    if mode == "constrained":
        body["max_tokens"] = 64
    out, ms = post(f"{a.llm_base_url.rstrip('/')}/chat/completions", body, {"authorization": f"Bearer {a.llm_api_key}"})
    text = out["choices"][0]["message"]["content"] or ""
    found = re.findall(r"\{[^{}]*\}", text)
    try:
        parsed = json.loads(found[-1]) if found else {}
    except json.JSONDecodeError:
        parsed = {}
    return {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in parsed.items()}, ms


def ask_dragonfly(record: dict, a) -> tuple[dict, float]:
    body = {"state": record["state"], "questions": {k: {f: v for f, v in q.items() if f in ("type", "instructions", "criteria")}
                                                    for k, q in record["questions"].items()}}
    headers = {"authorization": f"Bearer {a.dragonfly_key}"} if a.dragonfly_key else {}
    out, ms = post(f"{a.dragonfly.rstrip('/')}/v1/systemone", body, headers)
    answers = {}
    for qid, ans in out["answers"].items():
        if ans["type"] == "choice":
            answers[qid] = ans["choice"]
        elif ans["type"] == "noul":
            answers[qid] = "true" if ans["noul"] >= 0.5 else "false"
        else:
            answers[qid] = max(ans["probabilities"], key=ans["probabilities"].get)
    return answers, ms


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=50, help="requests sampled from --data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dragonfly", default="http://localhost:8000")
    ap.add_argument("--dragonfly-key", default=os.environ.get("DRAGONFLY_API_KEY"))
    ap.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL"))
    ap.add_argument("--llm-api-key", default=os.environ.get("LLM_API_KEY"))
    ap.add_argument("--llm-model", default=os.environ.get("LLM_MODEL"))
    ap.add_argument("--modes", default="constrained,reason")
    ap.add_argument("--out", help="write the full results as JSON here")
    a = ap.parse_args()

    records = [json.loads(line) for line in open(a.data, encoding="utf-8") if line.strip()]
    records = random.Random(a.seed).sample(records, min(a.n, len(records)))
    systems = [("dragonfly", lambda r: ask_dragonfly(r, a))]
    if a.llm_base_url and a.llm_model:
        systems += [(f"llm-{m}", lambda r, m=m: ask_llm(r, m, a)) for m in a.modes.split(",")]
    else:
        print("LLM_BASE_URL / LLM_MODEL not set: measuring Dragonfly only")

    results = {}
    for name, ask in systems:
        lat, right, total, errors = [], 0, 0, 0
        for r in records:
            try:
                answers, ms = ask(r)
            except Exception as e:
                errors += 1
                print(f"  {name}: request failed: {e}")
                continue
            lat.append(ms)
            for qid, q in r["questions"].items():
                total += 1
                right += answers.get(qid) == gold(q)
        results[name] = {"requests": len(lat), "errors": errors, "questions": total,
                         "accuracy": round(right / max(total, 1), 4),
                         "p50_ms": round(pct(lat, 50), 1) if lat else None, "p95_ms": round(pct(lat, 95), 1) if lat else None,
                         "mean_ms": round(statistics.mean(lat), 1) if lat else None}

    base = results["dragonfly"]["p50_ms"]
    print(f"\n{len(records)} requests from {a.data}" + (f", LLM = {a.llm_model}" if a.llm_model else ""))
    print(f"{'system':<18}{'accuracy':>9}{'p50 ms':>10}{'p95 ms':>10}{'p50 vs dragonfly':>18}")
    for name, s in results.items():
        ratio = f"{s['p50_ms'] / base:.1f}x slower" if base and s["p50_ms"] and name != "dragonfly" else "-"
        print(f"{name:<18}{s['accuracy']:>9.3f}{s['p50_ms'] or 0:>10.1f}{s['p95_ms'] or 0:>10.1f}{ratio:>18}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"llm_model": a.llm_model, "n": len(records), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
