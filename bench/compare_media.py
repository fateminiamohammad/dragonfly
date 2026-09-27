"""Dragonfly vs common autoregressive models on images and audio, same GPU, same inputs, measured the same way.

  images: Dragonfly (/v1/systemone with the image in the state: OCR + tags + decision, all one-pass models)
          vs a vision LLM via any OpenAI-compatible API (default: local Ollama qwen2.5vl:7b), in two modes:
            constrained  answer only (JSON), max 32 tokens, temperature 0   <- the fair comparison
            reason       think step by step, then answer                     <- how LLMs are often used
  audio:  Dragonfly CTC speech recognition (/v1/perceive) vs Whisper large-v3-turbo (autoregressive, transformers,
          run in-process on the same GPU; needs the perception-service venv).

  python bench/compare_media.py --dragonfly http://localhost:8000 --key <key> \
      --llm-base-url http://localhost:11434/v1 --llm-model qwen2.5vl:7b --whisper
"""

import argparse
import base64
import json
import statistics
import time
import urllib.request
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent.parent / "runs" / "samples"
KINDS = ["invoice", "photo of animals", "passport", "chart"]
IMAGE_TASKS = [
    # (file, question, options, correct)
    ("invoice.png", "What kind of document or picture is this?", KINDS, "invoice"),
    ("cats.jpg", "What kind of document or picture is this?", KINDS, "photo of animals"),
    ("invoice.png", "What is the total amount due?", ["$1,250.00", "$4,821.00", "$2,026.00", "$15.00"], "$1,250.00"),
]


def post(url, body, headers=None, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **(headers or {})})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t) * 1000


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def dragonfly_image(a, f, q, opts):
    body = {"state": {"attachment": {"type": "image", "data": b64(SAMPLES / f)}},
            "questions": {"q": {"type": "choice", "instructions": q, "criteria": {o: None for o in opts}}}}
    out, ms = post(f"{a.dragonfly}/v1/systemone", body, {"authorization": f"Bearer {a.key}", "cache-control": "no-cache"})
    return out["answers"]["q"]["choice"], ms


def llm_image(a, f, q, opts, mode):
    mime = "image/png" if f.endswith(".png") else "image/jpeg"
    prompt = (f"{q}\nOptions: {json.dumps(opts)}\n" +
              ("Think step by step, then on the last line output only JSON {\"answer\": <option>}." if mode == "reason"
               else "Output only JSON {\"answer\": <option>}. No other text."))
    body = {"model": a.llm_model, "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64(SAMPLES / f)}"}}]}]}
    if mode == "constrained":
        body["max_tokens"] = 32
    out, ms = post(f"{a.llm_base_url.rstrip('/')}/chat/completions", body, {"authorization": f"Bearer {a.llm_key}"})
    text = out["choices"][0]["message"]["content"] or ""
    answer = next((o for o in opts if f'"{o}"' in text[text.rfind("{"):]), None) if "{" in text else None
    return answer, ms


def run(label, fn, n):
    fn()  # warm-up (model load for the LLM, graph capture for Dragonfly)
    times, results = [], []
    for _ in range(n):
        ans, ms = fn()
        times.append(ms)
        results.append(ans)
    return {"system": label, "p50_ms": round(statistics.median(times), 1), "answers": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dragonfly", default="http://localhost:8000")
    ap.add_argument("--key", required=True)
    ap.add_argument("--llm-base-url", default="http://localhost:11434/v1")
    ap.add_argument("--llm-key", default="ollama")
    ap.add_argument("--llm-model", default="qwen2.5vl:7b")
    ap.add_argument("--n", type=int, default=5, help="timed runs per task")
    ap.add_argument("--whisper", action="store_true", help="also compare audio against Whisper large-v3-turbo")
    ap.add_argument("--out", default="runs/compare-media.json")
    a = ap.parse_args()
    report = {"images": [], "audio": []}

    print("== images: same question, same image")
    print(f"{'task':44} {'system':26} {'p50 ms':>9} {'correct':>8} {'vs dragonfly':>13}")
    for f, q, opts, gold in IMAGE_TASKS:
        rows = [run("dragonfly", lambda f=f, q=q, opts=opts: dragonfly_image(a, f, q, opts), a.n)]
        if a.llm_model:
            for mode in ("constrained", "reason"):
                rows.append(run(f"{a.llm_model} ({mode})",
                                lambda f=f, q=q, opts=opts, mode=mode: llm_image(a, f, q, opts, mode), a.n))
        base = rows[0]["p50_ms"]
        for r in rows:
            r["correct"] = sum(x == gold for x in r["answers"]) / len(r["answers"])
            ratio = "-" if r is rows[0] else f"{r['p50_ms'] / base:.0f}x slower"
            print(f"{f + ': ' + q[:30]:44} {r['system']:26} {r['p50_ms']:9.1f} {r['correct']:8.0%} {ratio:>13}")
        report["images"].append({"file": f, "question": q, "gold": gold, "results": rows})

    print("\n== audio: speech -> text, 10.4 s LibriSpeech clip")
    audio = {"items": [{"id": "a", "type": "audio", "data": b64(SAMPLES / "speech.flac"), "language": "en"}]}
    rows = [run("dragonfly CTC (parakeet-ctc-0.6b)",
                lambda: (lambda o: (o[0]["items"]["a"]["transcript"], o[1]))(
                    post(f"{a.dragonfly}/v1/perceive", audio, {"authorization": f"Bearer {a.key}"})), a.n)]
    if a.whisper:
        import torch
        from perception.audio import decode_audio
        from transformers import pipeline
        wav = decode_audio((SAMPLES / "speech.flac").read_bytes(), 60)
        asr = pipeline("automatic-speech-recognition", "openai/whisper-large-v3-turbo", device=0,
                       dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

        def whisper():
            t = time.perf_counter()
            text = asr({"raw": wav, "sampling_rate": 16000})["text"]
            torch.cuda.synchronize()
            return text.strip().lower().replace(",", "").replace(".", ""), (time.perf_counter() - t) * 1000
        rows.append(run("whisper-large-v3-turbo (autoregressive, in-process)", whisper, a.n))
    base = rows[0]["p50_ms"]
    for r in rows:
        ratio = "-" if r is rows[0] else f"{r['p50_ms'] / base:.1f}x slower"
        print(f"{r['system']:52} {r['p50_ms']:9.1f} ms {ratio:>13}  | {r['answers'][0][:70]}")
    report["audio"] = rows
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
