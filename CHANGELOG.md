# Changelog

All notable changes to this project. The plugin API (`dragonfly.plugins.api.API_VERSION`) follows semver.

## [Unreleased]: v0.2.0

### Plugins
- Plugin API 1.1: `blocking = True` plugins run their hooks in a thread pool with a `timeout_s` budget, so network I/O
  never stalls the event loop. gRPC plugins are blocking.
- Free built-in plugins (`dragonfly.essentials`): `guardrails` (PII redaction, injection/profanity checks, state size
  limit), `human-review` (Redis queue of unsure answers), `webhook-audit` (webhook / Slack / JSONL with filters and
  retries) and `llm-escalation` (any OpenAI-compatible LLM for unsure answers, marked `tier: "llm"`).
- Backend `/api/review` and a UI **Review** page: label unsure decisions and export them as training JSONL.

### Data
- `scripts/build_mix.py`: the mix-v1 training set (32.9k questions), deduplicated and screened against every test split.

## v0.1.0

### Model
- Dragonfly-S: ModernBERT encoder + pointer head; one row per question, options placed before the state so truncation
  never cuts an option.
- Dragonfly-M: Qwen3 + LoRA, one packed sequence per request. The state is read once, questions are isolated, and every
  option starts at the same position, so reordering options cannot change the scores (tested to within 1e-5).
- Cascade S → M for questions below a confidence threshold.
- Training (`dragonfly-train`) with option-order augmentation and temperature scaling. Evaluation (`dragonfly-eval`)
  reports accuracy, ECE, Brier, NLL, the share automatable at 5% error, and option-order flip rate.

### Service
- TypeSafe-compatible `/v1/systemone` (plus `/v1/decide`, `/v1/models`, `/health`, `/metrics`).
- Dynamic batching, answer cache, optional `torch.compile`.
- API keys: static, or issued by the backend and shared through Redis; usage counted per key and day.

### Plugins (API 1.0)
- In-process Python plugins via the `dragonfly.plugins` entry point: `on_request`, `on_decision`, `on_low_confidence`.
- Out-of-process gRPC sidecar plugins in any language (`proto/dragonfly/plugin/v1/plugin.proto`), with a time budget
  and a fail-open or fail-closed policy per plugin.

### Images and audio (perception-service)
- New microservice that turns media into text with **non-autoregressive** models only:
  - speech → text: CTC (`parakeet-ctc-0.6b` for English, `wav2vec2-xlsr-53-persian` for Persian);
  - text in images: PP-OCRv4 via RapidOCR (torch engine);
  - what an image shows: SigLIP-2 against a 216-tag vocabulary.
- Media objects (`{"type": "image" | "audio", "data": base64}`) anywhere in a request's `state` are converted in one
  batched call before plugins and the model run. `POST /v1/perceive` converts without deciding.
- UI: a Media page, and an "Attach image/audio" button in the Playground. The backend proxies `/api/model/perceive`
  with a 30 MB JSON limit.
- Measured on an RTX 3090 Ti: 10.4 s of speech → exact transcript in 79 ms; an invoice → exact OCR in 66 ms; tags in
  11 ms (CUDA graphs).
- Fixes found on real data:
  - fp16 overflowed Parakeet-0.6B, so bf16 is used;
  - RapidOCR's default resize doubled detection time, so the longer side is capped at 960 px;
  - SigLIP sigmoid probabilities aren't calibrated for a large vocabulary, so tags are ranked by a relative share.

### Tier M and cascade
- Dragonfly-M trained: 76.9% accuracy, ECE 0.027, option-order flip rate 0.86%.
- Cascade threshold 0.55, tuned on the calibration split (`scripts/tune_cascade.py`): **77.5%** test accuracy with
  42% of questions escalated.
- CUDA graphs for tier M: one request 113 → 21 ms. Fine token buckets, since decoder padding is real work.
  Common shapes are captured at startup (`DRAGONFLY_WARMUP`).
- State (document) KV cache for tier M, used for states of 512 tokens or more. It is tested identical to the full pass,
  including a mutation test.
- Knowledge distillation from M into S (`scripts/distill_targets.py`: soft targets are a label/teacher mix). The
  distilled S improves on every metric: 68.2% accuracy, ECE 0.032, auto_rate@5% 39.4%.
- `Cache-Control: no-cache` skips the answer-cache lookup, so benchmarks measure the model.

### Benchmarks vs common LLMs (same GPU, local Ollama, answer cache bypassed)
- Text, `bench/compare_llm.py`, served cascade:
  - Dragonfly: 79.1% at 18 ms p50;
  - Qwen2.5-7B answering directly: 74.4% at 103 ms (5.7× slower);
  - Qwen2.5-7B reasoning first: 76.7% at 995 ms (55× slower);
  - thinking Qwen3-8B: 80.0% at 1,658 ms against Dragonfly's 84.6% at 16 ms on the same 50 requests (**105× slower**).
- Images and audio, `bench/compare_media.py`:
  - image decisions in 71–104 ms, against Qwen2.5-VL-7B at 2.2–3.0 s (22–39×);
  - 10.4 s of speech in 71 ms, against Whisper large-v3-turbo at 551 ms (7.8×), with identical words.
- 200× against a hosted frontier API has not been measured (it needs a key).

### Benchmark vs Kev and Jev
- `bench/compare_systemone.py` runs any number of `/v1/systemone` servers on identical labelled requests. Jev is
  included when `TYPESAFE_API_KEY` is set.
- Kev is served from its repo at a pinned commit (`bench/kev/Dockerfile`, compose profile `kev`).
- Results on the same GPU and 200 test requests:
  - Dragonfly: 77.1% at 22 ms p50;
  - Kev-0.8B: 83.1% at 43 ms;
  - Kev-4B: 89.2% at 44 ms.
- So Dragonfly is about 2× faster (about 3× at 8 clients) and less accurate. Kev never trained on these test questions
  (checked against every data file in its repo).

### Benchmark vs Laya, and Jev's published numbers
- Laya is served from `laya[serve]==0.3.21` (`bench/laya/`, compose profile `laya`).
- The typed-decisions benchmark is converted with `scripts/convert_typed_decisions.py`.
- typed-decisions test (400 cases):
  - Laya-typed-decisions: 76.7% at 55 ms;
  - Dragonfly-td (tier S trained on its train split): 73.1% at 47 ms;
  - Kev-4B zero-shot: 66.9% at 110 ms;
  - Dragonfly general, zero-shot: 46.1%;
  - Jev (published leaderboard): 72.7% at 710 ms hosted.
- decision-v2: Laya base 66.3% at 48 ms.
- Jev's published benchmarks are collected with sources in `docs/MODEL.md`.

### Serving
- Tier M: LoRA merged into the weights at load (identical output). CUDA-graph buckets up to 64 rows. Warm-up capture
  at startup, which took p95 from 365 to 38 ms.
- Distilled tier S served with cascade threshold 0.45: 76.6% test, 35.5% escalated.
- The perception-service warms its models at startup (the first request took 3.7 s before).

### Performance
- CUDA graphs for tier S (`DRAGONFLY_CUDA_GRAPHS`, on by default on GPU). Inputs are padded to shape buckets. Measured
  model time went from 30 to 8.5 ms per request on an RTX 3090 Ti; results match eager to 1.2e-5 in fp32.

### Fixes found by training on real data
- The temperature fit returned NaN when questions had different option counts (-inf padding). Now padded with a finite
  value, with a guard and a regression test.
- Banking77 questions with 77 described options exceed 1,024 tokens. The tier S default is now 1,536, and training
  skips any question that still does not fit instead of crashing.
- The model-service image needs a C compiler for Triton (ModernBERT on CUDA, `torch.compile`).
- New `dragonfly-calibrate` command: re-fits the temperature of a saved checkpoint without retraining.

### Platform
- Backend (NestJS, Postgres, Redis): accounts, API keys, usage, model info, playground.
- UI (React): playground, API keys, usage, model and plugins.
- Docker compose for local, dev and production; nginx edge with TLS; deploy script with automatic rollback;
  Prometheus/Grafana profile.
- CI for all services; images published to GHCR.
- Benchmarks: latency (`bench/latency.py`) and Dragonfly vs LLM, fair and unfair modes (`bench/compare_llm.py`).
