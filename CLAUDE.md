# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository. Keep it current: when a change affects what's
summarized here, update this file in the same commit.

## Summary

**Dragonfly** is an open-source (Apache-2.0) **System One decision model**, an open alternative to TypeSafe's Jev
that builds on ideas from Kev. You send a document plus typed questions (`choice` / `noul` / `score`); it returns
calibrated probabilities for every answer in **one forward pass**, with no text generation. That's why it answers in
milliseconds instead of an LLM's seconds.

It speaks the TypeSafe-compatible `POST /v1/systemone` API. It ships as Docker microservices with an open plugin
system, and plugins may be closed source:
- AI model;
- perception (images and audio → text);
- backend;
- UI.

Every model in the request path is **non-autoregressive**: one forward pass, with no token-by-token generation.

- **Repo:** https://github.com/fateminiamohammad/dragonfly (branch `main`)
- **Images:** `ghcr.io/fateminiamohammad/dragonfly-*`

## Current status (2026-09-28)

| Area | State |
|---|---|
| Dragonfly-S2 (ModernBERT-base, speed tier) | trained on mix-v1 (32.9k questions), **distilled from M-4B**: 80.1% decision-v2, ECE 0.063; serving `runs/dragonfly-s2` |
| Dragonfly-M (Qwen3-4B + LoRA, quality tier, `head_norm`) | trained on mix-v1: 86.2% decision-v2, 79.3% typed-decisions, ECE 0.047, 0.14% order flips; serving `runs/dragonfly-m4b` (LoRA merged unless tier M specialists are installed) |
| Cascade S→M | threshold 0.70 (tuned on mix-v1 calibration): **84.4% decision-v2 test**, 26.2% escalated; 0.45 = fast point (82.3%, 12.3 ms p50) |
| Latency (RTX 3090 Ti, CUDA graphs on both tiers, warm-up at startup) | tier S ~5 ms; 3-question request with 2 escalated to M-4B: 49 ms end to end; 32 clients: 31 req/s. Idle GPU downclocks: first request after a pause 200–350 ms unless clocks are locked |
| vs local LLMs, same GPU (`bench/compare_llm.py`) | more accurate than all; 5.7× vs Qwen2.5-7B answer-only, 55× vs it reasoning, **105× vs thinking Qwen3-8B** |
| Images and audio (perception-service) | done: CTC speech (en/fa), PP-OCR, SigLIP-2 tags; 22–39× faster than Qwen2.5-VL-7B, 7.8× faster than Whisper-turbo |
| Backend, UI (Media page), keys and usage via Redis, plugins (Python + gRPC) | done, verified end to end (`scripts/e2e_media.py`) |
| CI and GHCR publishing | green |
| vs Kev / Laya (same GPU, identical requests; `bench/compare_systemone.py`, `runs/v02-*.json`) | decision-v2: Dragonfly 84.6% at 16.5 ms p50 vs Kev-4B 87.9% at 43 ms, Kev-0.8B 85.0% at 16.7 ms, Laya 64.4%; Dragonfly best p95/throughput. typed-decisions: **Dragonfly most accurate (79.2%)** vs Laya-td 76.8%, Jev (published) 72.7%, Kev-4B 67.0%; but 125 ms p50 vs Laya-td 57 ms |
| vs Jev | measured only via **published** numbers (typed-decisions 72.7% at 710 ms hosted); direct run needs `TYPESAFE_API_KEY` |
| Swarm (v0.2) | specialists by `model` name or `auto`; tier M adapters switch in 2.4 ms; train-your-own from the UI (trainer-worker); agent flows (`/v1/flows`); free plugins (guardrails, human-review, webhook-audit, llm-escalation); scale-out behind nginx `least_conn` |
| **200× vs GPT** | **not measured**: no hosted-API key. Local thinking LLM measured at 105×. |

**Next steps** (v0.2 plan, `docs/SWARM.md`):
0. Phase A2/A3: S-large (ModernBERT-large) distilled from M-4B to cut escalations (typed-decisions escalates 71%);
   weight-only INT8 for M-4B. Adopt only with measured gains.
1. Phase A4: broader data (more public suites with per-source caps) and 2 epochs for M-4B; `docs/TRAINING.md`.
2. Phase F: "layers of each model" diagrams in `docs/ALGORITHM.md`.
3. Measure against a hosted frontier API (set `LLM_*`, run `bench/compare_llm.py`); publish checkpoints.

## Tech stack

- **model-service:** Python 3.12, PyTorch, Transformers, FastAPI/uvicorn, Redis client, httpx, gRPC (`./model-service/`)
- **perception-service:** Python 3.12, PyTorch, Transformers (Parakeet/wav2vec2 CTC, SigLIP-2), RapidOCR, soundfile, ffmpeg (`./perception-service/`)
- **backend:** NestJS 10, TypeORM, Postgres 16, ioredis, JWT (`./backend/`)
- **ui:** React 18, Vite, TypeScript (`./ui/`)
- **infra:** Docker Compose, nginx edge (TLS via certbot in prod), Redis 7, Prometheus/Grafana (prod profile)

## Services and ports (local)

| Service | Port | Notes |
|---|---|---|
| nginx edge | **8080** | one origin: `/` → ui, `/api` → backend, `/v1` + `/health` → model-service |
| model-service | 8000 | also direct; `/metrics` for Prometheus |
| perception-service | 8001 (internal) | reached through model-service `/v1/perceive` or media in the state |
| llm (Ollama) | 127.0.0.1:11434 | `llm` profile, only for benchmarks (qwen2.5:7b-instruct, qwen2.5vl:7b, qwen3:8b pulled) |
| kev | 127.0.0.1:8009 | `kev` profile, benchmark opponent; `KEV_MODEL=jaredpalmer/kev-0.8b|kev-4b|kev-9b`; call it via `127.0.0.1`, not `localhost` |
| laya | 127.0.0.1:8010 | `laya` profile, benchmark opponent; the request's `model` picks `english` / `multilingual` / `typed-decisions` |
| backend | 3000 (internal) | reached through nginx `/api` |
| ui | 80 (internal) | reached through nginx `/` |
| postgres | 5434 → 5432 | |
| redis | internal | API-key set and usage counters |
| plugin-policy | 50051 (internal) | example gRPC plugin, `plugins` profile |

The local login is `ADMIN_EMAIL` / `ADMIN_PASSWORD` from `docker/.env`.

## Commands

```bash
# run everything
docker compose -f docker-compose.local.yml --env-file docker/.env up -d --build
docker compose -f docker-compose.local.yml --env-file docker/.env --profile plugins up -d   # + gRPC example plugin
docker compose -f docker-compose.local.yml --env-file docker/.env down

# train / calibrate (in Docker; see docs/MODEL.md)
docker compose -f docker-compose.local.yml --env-file docker/.env --profile train up -d trainer
MSYS_NO_PATHCONV=1 docker compose -f docker-compose.local.yml --env-file docker/.env --profile train run --rm \
  --entrypoint python trainer -m dragonfly.training.calibrate --checkpoint /runs/dragonfly-s \
  --calibration /data/decision-v2/calibration.jsonl --test /data/decision-v2/test.jsonl

# tests (all must pass before committing)
cd model-service && .venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff check --config pyproject.toml src tests ../plugins ../bench ../scripts
cd backend && npm run typecheck && npm run lint && npm test && npm run build
cd ui && npm test && npm run build

# benchmarks (Dragonfly requests bypass the answer cache)
python bench/latency.py --url http://localhost:8000 --api-key <key> --concurrency 1
python bench/compare_llm.py --data data/decision-v2/test.jsonl --n 50 --dragonfly http://localhost:8000
python bench/compare_media.py --key <key> --llm-model qwen2.5vl:7b --whisper      # perception venv
python scripts/tune_cascade.py --small runs/dragonfly-s2 --large runs/dragonfly-m4b --calibration data/mix-v1/calibration.jsonl --test data/decision-v2/test.jsonl --out runs/cascade.json
python scripts/e2e_media.py --password <ADMIN_PASSWORD>                            # full media path through nginx
```

## Repository map

| Path | Contents |
|---|---|
| `model-service/src/dragonfly/schema.py` | `/v1/systemone` request and response shapes; confidence formulas |
| `model-service/src/dragonfly/models/encoder.py` | tier S (ModernBERT + pointer head) |
| `model-service/src/dragonfly/models/decoder.py` | tier M (Qwen3 + LoRA, packing, attention mask) |
| `model-service/src/dragonfly/models/graphs.py` | CUDA graphs: `GraphRunner` (tier S), `DecoderGraphRunner` (tier M) |
| `model-service/src/dragonfly/media.py` | media objects in the state → perception-service → text |
| `perception-service/src/perception/` | `audio.py` (CTC ASR), `ocr.py` (PP-OCR, RTL reading order), `tags.py` (SigLIP-2), `api.py` |
| `model-service/src/dragonfly/engine.py` | load/save checkpoints, `Engine.probs`, `Cascade` |
| `model-service/src/dragonfly/batching.py`, `cache.py`, `auth.py` | model worker thread, answer cache, API keys and usage via Redis |
| `model-service/src/dragonfly/api/` | FastAPI app (`app.py`) and entry point (`serve.py`, env config) |
| `model-service/src/dragonfly/plugins/` | plugin API (`api.py`), host, gRPC client (`grpc_plugin.py`, `wire.py`) |
| `model-service/src/dragonfly/training/` | `train.py`, `evaluate.py`, `calibrate.py` |
| `backend/src/` | auth, keys, usage, model proxy, entities, migrations |
| `ui/src/` | pages: Playground, Keys, Usage, Model; `api.ts`, `format.ts` |
| `proto/dragonfly/plugin/v1/plugin.proto` | gRPC plugin protocol |
| `plugins/example-python`, `plugins/example-grpc` | example plugins |
| `bench/`, `scripts/` | benchmarks; `fetch_data.py`, `distill_targets.py`, `tune_cascade.py`, `e2e_media.py`, `publish_hf.py` |
| `docker-compose.{local,dev,prod}.yml`, `docker/.env.example`, `nginx/`, `ops/` | deployment |
| `data/`, `runs/` | datasets and checkpoints (gitignored) |

## Rules

- **Never commit secrets.** Every `.env`, `.env.*` and `*.env` file is gitignored except `.env.example`. New settings
  go into `docker/.env.example` with a safe placeholder.
- **Numbers need evidence.** Don't claim accuracy or speed without output from `dragonfly-eval`, `dragonfly-calibrate`
  or `bench/*`, naming the hardware. Benchmarks must bypass the answer cache (the default in `bench/latency.py`).
- **Keep `/v1/systemone` TypeSafe/Kev-compatible.** Add fields; never rename or remove them.
- **The plugin API is semver** (`dragonfly.plugins.api.API_VERSION`). Breaking changes need a major version bump and
  a CHANGELOG entry.
- **Keep the hot path fast.** The backend is never on the `/v1` path. No blocking I/O in request handlers. Usage
  writes happen after the response.
- **Tier M must stay order-invariant.** `tests/test_decoder.py` guards it; don't change packing or masks without it
  passing.
- **Schema changes go through TypeORM migrations** in `backend/src/migrations/`. Never use `synchronize`.
- **Commits are authored as** `Mohammad Fateminia <fateminiamohammad@users.noreply.github.com>` (set in this repo's
  git config).

## Gotchas

- **Git Bash path rewriting:** prefix docker commands that pass container paths (`/runs/...`) with
  `MSYS_NO_PATHCONV=1`.
- **Bind-mount reload is unreliable on Windows:** after editing source, `docker restart dragonfly-backend` or
  `dragonfly-model-service`.
- **Run long GPU training through the Docker `trainer` services,** not from the session shell.
- **`DRAGONFLY_API_KEYS` must include `MODEL_SERVICE_API_KEY`,** or the UI playground gets 401.
- **An enabled gRPC plugin that's unreachable stops model-service startup.** That's deliberate: fail closed.
- **The GPU is shared.** Serving (S + M + perception) and Ollama together fill most of 24 GB. Stop `dragonfly-llm`
  before training, or training crawls on memory pressure (seen once: a trainer stuck at step 50).
- **Use bf16, not fp16, for the perception models.** Parakeet-0.6B overflowed to NaN in fp16.
- **Benchmarks must bypass the answer cache** (`Cache-Control: no-cache`, or unique states). An early run reported
  3.4 ms that was actually cache hits.

## Documentation

- [README.md](README.md): overview, quick start, first results
- [docs/ALGORITHM.md](docs/ALGORITHM.md): how it works, visually (Mermaid diagrams plus the tier M mask SVG)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): services, hot path, model design, cascade, calibration
- [docs/MODEL.md](docs/MODEL.md): tiers, data, training/calibration/eval commands, metrics, results, speed
- [docs/API.md](docs/API.md): `/v1` and `/api` endpoints, request and response shapes, errors, key and usage flow
- [docs/PERCEPTION.md](docs/PERCEPTION.md): images and audio, models, API, limits, measured speed
- [docs/PLUGINS.md](docs/PLUGINS.md): writing in-process and gRPC plugins, closed-source delivery
- [docs/SWARM.md](docs/SWARM.md): specialist dragonflies, adapter switching, `model: "auto"` routing
- [docs/LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md): running with and without Docker, tests, Windows gotchas
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): production server, TLS, deploys with rollback, monitoring, security
- [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md)
