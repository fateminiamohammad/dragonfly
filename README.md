# Dragonfly

**An open-source System One decision model.** You send a document and typed questions; you get back calibrated
probabilities for every possible answer, in **one forward pass**. Dragonfly never generates text, which is why it can
answer in milliseconds where an LLM takes seconds.

Dragonfly is an open alternative to [TypeSafe's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
and builds on ideas from [Kev](https://github.com/jaredpalmer/kev). It speaks the same `/v1/systemone` API, so existing
TypeSafe and Kev clients work by changing `base_url`.

> **Status: pre-release (v0.1.0.dev).** The whole platform is built and tested: model, API, plugins, backend, UI, Docker
> and CI. Dragonfly-S has been trained and measured (below). Dragonfly-M is built and tested but not yet trained, and
> checkpoints are not published yet.

## What it does

```jsonc
// POST /v1/systemone
{
  "state": "I ordered a new card two weeks ago and it still hasn't arrived.",
  "questions": {
    "intent":   { "type": "choice", "criteria": { "card_arrival": null, "card_not_working": null, "change_pin": null } },
    "escalate": { "type": "noul",   "instructions": "Escalate to a human now?" },
    "mood":     { "type": "score",  "criteria": ["very negative", "negative", "neutral", "positive", "very positive"] }
  }
}
// -> every question answered in the same pass, each with calibrated probabilities
{
  "answers": {
    "intent":   { "type": "choice", "choice": "card_arrival", "confidence": 0.93, "probabilities": { ... }, "tier": "S" },
    "escalate": { "type": "noul", "noul": 0.21, "confidence": 0.58, "tier": "S" },
    "mood":     { "type": "score", "score": 1.12, "confidence": 0.74, "probabilities": { ... }, "tier": "M" }
  },
  "latency_ms": 4.8
}
```

| Type | Answer |
|---|---|
| `choice` | one of up to 255 named options, with a probability for each |
| `noul` | a calibrated yes/no probability |
| `score` | the expected level on an ordered scale, with a probability per level |

## First results (Dragonfly-S, measured)

**Setup:**
- Dragonfly-S: ModernBERT-base, 150M parameters.
- Trained 3 epochs (6 minutes) on Kev's decision-v2 suite.
- Measured on its 1,440 held-out test questions and an RTX 3090 Ti.

| Metric | Result |
|---|---|
| Accuracy | 67.7% (noul 79.0%, choice 64.8%, score 52.5%) |
| Calibration error (ECE, lower is better) | 0.038 |
| Share automatable at 5% error | 35.6% |
| Option-order flip rate | 13.4% (tier S sees option order; tier M is order-invariant by construction) |
| Latency, 1 client, 3-question request | 14.1 ms p50 end to end (8.5 ms model) |
| Throughput, 32 clients | 153 requests/s (460 questions/s) |

**Context:**
- This is a small first model. Kev-4B reports 85.6% in-distribution with a 4B model.
- Dragonfly-M (Qwen3 + LoRA) is the tier meant to close that gap. It is not trained yet.
- **Latency is still above the 5 ms target.** The benchmark vs LLM APIs has not been run yet; it needs an API key.

Reproduce: `dragonfly-train` / `dragonfly-calibrate`, then `bench/latency.py` (see below).

## Models

| Tier | Model | Role |
|---|---|---|
| **S** | ModernBERT encoder + pointer head | Speed: answers every question first |
| **M** | Qwen3 + LoRA + pointer head, one packed pass per request | Quality: re-answers only questions S is unsure about |

Dragonfly-M reads the document once and keeps questions and options isolated from each other. Every option also starts
at the same position, so **reordering options cannot change the answer**. This holds by construction and is tested to
within 1e-5; Kev changes its answer under reordering about 7% of the time.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Why it is fast

1. **One forward pass, no generation:** every option is scored in parallel.
2. **Many questions per request:** they share one pass instead of one LLM call each.
3. **Cascade:** the small tier S answers most questions; only unsure ones pay for tier M.
4. **CUDA graphs:** the tier S forward pass replays as one GPU launch instead of about 680 (30 → 8.5 ms measured).
5. **Local serving, dynamic batching, answer cache, optional `torch.compile`.**

**Target:** about 5 ms per request on an RTX 3090 Ti (measured today: 14 ms; see above). That is about 200× faster than an LLM that writes reasoning plus
JSON (1 s or more), and about 20–50× faster than an LLM constrained to output only the answer.

`bench/compare_llm.py` measures **both** comparisons on the same data against any OpenAI-compatible API, so you can
check these numbers yourself.

## Architecture

```
                      ┌─────────── nginx (TLS) ───────────┐
 clients / SDKs ────► │ /v1/*  → model-service (hot path) │
 browser ───────────► │ /api/* → backend                  │
                      │ /      → ui                       │
                      └───────────────────────────────────┘
 model-service  Python/PyTorch · tiers S+M · batching · cache · plugins · /metrics
 backend        NestJS · Postgres · accounts, API keys, usage, playground
 ui             React · playground, keys, usage, model & plugins
 redis          API keys and usage, shared by backend and model-service
 plugins        Python packages (in-process) or gRPC sidecars (any language)
```

The backend is **never on the inference path**: `/v1` goes straight to the model-service, which checks API keys against
an in-process cache backed by Redis.

## Quick start

With Docker and an NVIDIA GPU:

```bash
cp docker/.env.example docker/.env        # then change the passwords and keys
docker compose -f docker-compose.local.yml --env-file docker/.env up --build
```

- **UI:** open http://localhost:8080 and sign in with `ADMIN_EMAIL` / `ADMIN_PASSWORD`.
- **API:** create a key in the UI, then call the model:

```bash
curl localhost:8080/v1/systemone -H "authorization: Bearer <key>" -H 'content-type: application/json' \
  -d @model-service/examples/request.json
```

No GPU? Set `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu` and `DRAGONFLY_DEVICE=cpu` in `docker/.env`.

### Train

Training uses Kev's decision-v2 suite: 10 public datasets, the same data Kev reports on.

```bash
python scripts/fetch_data.py
docker compose -f docker-compose.local.yml --env-file docker/.env --profile train   run --rm trainer     # tier S
docker compose -f docker-compose.local.yml --env-file docker/.env --profile train-m run --rm trainer-m   # tier M
```

Then set `DRAGONFLY_CHECKPOINT=/runs/dragonfly-s` (and `DRAGONFLY_CHECKPOINT_M=/runs/dragonfly-m` to turn on the
cascade) and restart. Each run writes `metrics.json`: accuracy, ECE, Brier, the share automatable at 5% error, and the
option-order flip rate. `scripts/publish_hf.py` turns those measurements into a Hugging Face model card.

### Benchmark

```bash
python bench/latency.py --url http://localhost:8000 --concurrency 32            # p50/p95 and throughput
LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=... \
  python bench/compare_llm.py --data data/decision-v2/test.jsonl --n 50        # vs an LLM, fair and unfair modes
```

## Plugins

The core is Apache-2.0. Plugins are separate, so they can be **open or closed source**.

**Two kinds:**
- **Python packages**, loaded in-process through an entry point, with microsecond overhead.
- **gRPC sidecars** in any language, with their own container, a time budget, and a fail-open or fail-closed policy.

**Hooks:**
- `on_request`: before the model;
- `on_decision`: after the model;
- `on_low_confidence`: for unsure answers, e.g. escalation to an LLM or a human.

See [docs/PLUGINS.md](docs/PLUGINS.md), [plugins/example-python](plugins/example-python) and
[plugins/example-grpc](plugins/example-grpc).

## Repository

| Path | Contents |
|---|---|
| `model-service/` | Python package `dragonfly`: models, engine, API, plugins, training, tests |
| `backend/` | NestJS API: auth, keys, usage, model proxy |
| `ui/` | React dashboard |
| `proto/` | gRPC plugin protocol |
| `plugins/` | example plugins (Python and gRPC) |
| `bench/`, `scripts/` | benchmarks, data download, Hugging Face publishing |
| `nginx/`, `ops/`, `docker/` | edge proxy, deploy/TLS/monitoring, environment template |
| `docs/` | architecture, plugins, deployment |

## Roadmap

- [x] **M0** repository, Docker, CI
- [x] **M1** model-service: `/v1/systemone`, Dragonfly-S, batching, calibration, training and evaluation
- [x] **M2** Dragonfly-M: Qwen3 + LoRA, packed pass, order-invariant options
- [x] **M3** cascade S→M, answer cache, `torch.compile`, benchmarks vs LLMs
- [x] **M4** backend (NestJS) and UI (React), API keys and usage through Redis
- [x] **M5** gRPC sidecar plugins in any language
- [ ] **M6** v0.1 release: trained checkpoints on Hugging Face, images on GHCR, published benchmark report
- [ ] Next: a state KV cache for tier M across requests, ONNX/TensorRT export for tier S, and RLCD-style calibration
      training

## License

[Apache-2.0](LICENSE). Training data from Kev's suites keeps each source dataset's own license.
