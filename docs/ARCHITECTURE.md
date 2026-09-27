# Architecture

```
                    ┌──────────── nginx (edge, TLS) ─────────────┐
 clients / SDKs ──► │ /v1/*  → model-service   (hot path, direct)│
                    │ /api/* → backend                           │
                    │ /      → ui                                │
                    └────────────────────────────────────────────┘
 model-service (Python, GPU)
   ├─ API (FastAPI): /v1/systemone, /v1/decide, /v1/models, /health, /metrics
   ├─ auth: static keys + backend-issued keys from Redis (cached in-process); usage counters to Redis
   ├─ plugin host: in-process plugins (entry points) and gRPC sidecars
   ├─ answer cache: repeated requests skip the GPU
   ├─ worker thread: dynamic batching, one forward pass per batch
   └─ engine: Dragonfly-S (encoder), Dragonfly-M (decoder + LoRA), or the S -> M cascade
 backend (NestJS, Postgres): accounts, API keys (-> Redis), usage (<- Redis), model info, playground
 ui (React): playground, keys, usage, model & plugins
 trainer (same image, profile: train) → checkpoints in ./runs
```

**The hot path is only nginx → model-service.** The backend manages API keys and reads usage. The two share
state only through Redis, and a key check costs no network round trip after the first request. The backend is never
in the request path, so it adds no latency.

## Model

Every question type reduces to **one primitive: a pointer over options**.
- `noul` is a choice between "no" and "yes".
- `score` is a choice over ordered levels; the answer is the expected level.

### Dragonfly-S (`model-service/src/dragonfly/models/encoder.py`)

Each question is one row:

```
[CLS] instructions  - option 1  - option 2 ...  [SEP] state [SEP]
```

- A bidirectional encoder (ModernBERT-base by default) reads the row.
- The pointer head compares the `[CLS]` query with each option's mean-pooled tokens, using a scaled dot product, then
  a softmax over the options.
- Options come before the state, so truncation only ever cuts the state.
- The questions in a request, and the requests in a batch, run as rows of one forward pass.

### Dragonfly-M (`model-service/src/dragonfly/models/decoder.py`)

- Qwen3 (1.7B by default), frozen, plus LoRA adapters (r=16) and the same pointer head.
- A whole request is **one packed sequence**: `[state][question 1][options 1...][question 2][options 2...]`.
- A custom attention mask means:
  - the state is read once and shared by every question;
  - questions never see each other;
  - options never see each other.
- Every option of a question starts at the **same position ID**. Reordering options therefore cannot change their
  scores. This is order invariance by construction, verified in `tests/test_decoder.py` to within 1e-5.
- A checkpoint stores only the LoRA adapters and the head. The base model comes from the Hub.

Train it with `dragonfly-train --tier M` or the `trainer-m` compose profile.

### Cascade (`engine.Cascade`)

S answers first. Questions whose calibrated confidence is below `DRAGONFLY_CASCADE_THRESHOLD` go to M, packed per
request so M reads each state once. Each answer reports the `tier` that produced it. Questions M is still unsure about
can go to `on_low_confidence` plugins.

## Calibration

1. Train with cross-entropy (a proper scoring rule), against soft targets when available.
2. Fit a temperature on a held-out calibration split.

`dragonfly-eval` reports:
- accuracy, ECE, Brier and NLL;
- the share of decisions automatable at a 5% error budget;
- the option-order flip rate.

## Why not C++, Fortran or assembly?

The heavy math already runs in NVIDIA's tuned C++/CUDA kernels. Python only orchestrates it. Speed comes from:
- the design: one pass, no generation, batching, caching;
- the model size;
- the inference runtime: `torch.compile` today; TensorRT/ONNX export is on the roadmap.

A native rewrite pays off only for the last few hundred microseconds of HTTP overhead, if profiling ever shows that
matters.
