# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository. Keep it current: when a change affects what's
summarized here, update this file in the same commit.

## Summary

**Dragonfly** is an open-source (Apache-2.0) **System One decision model**, an open alternative to TypeSafe's Jev
that builds on ideas from Kev. You send a document plus typed questions (`choice` / `noul` / `score`); it returns
calibrated probabilities for every answer in **one forward pass**, with no text generation. That's why it answers in
milliseconds instead of an LLM's seconds.

It speaks the TypeSafe-compatible `POST /v1/systemone` API. It ships as Docker microservices (AI model / backend / UI)
with an open plugin system, and plugins may be closed source.

- **Repo:** https://github.com/fateminiamohammad/dragonfly (branch `main`)
- **Images:** `ghcr.io/fateminiamohammad/dragonfly-*`

## Current status (2026-09-27)

| Area | State |
|---|---|
| Dragonfly-S (ModernBERT, speed tier) | **trained**: 67.7% accuracy, ECE 0.038 on decision-v2 test; serving from `runs/dragonfly-s` |
| Dragonfly-M (Qwen3 + LoRA, quality tier) | built and tested (order-invariant by construction); **not trained yet** |
| Latency (RTX 3090 Ti, CUDA graphs) | 14.1 ms p50 end to end, 8.5 ms model, for a 3-question request; target is 5 ms |
| Cascade S→M, answer cache, batching | done |
| Backend, UI, Redis-shared keys and usage | done, verified end to end |
| Plugins (Python in-process + gRPC sidecars) | done; example `policy` sidecar running |
| CI (GitHub Actions) and GHCR publishing | green |
| Benchmark vs LLM APIs | script ready (`bench/compare_llm.py`); **not run**, needs an LLM API key |

**Next steps:**
1. Train tier M.
2. Run the LLM comparison.
3. Add a KV cache of the state for tier M.
4. Add ONNX/TensorRT export for tier S.
5. RLCD-style calibration training.
6. Publish checkpoints to Hugging Face.

## Tech stack

- **model-service:** Python 3.12, PyTorch, Transformers, FastAPI/uvicorn, Redis client, gRPC (`./model-service/`)
- **backend:** NestJS 10, TypeORM, Postgres 16, ioredis, JWT (`./backend/`)
- **ui:** React 18, Vite, TypeScript (`./ui/`)
- **infra:** Docker Compose, nginx edge (TLS via certbot in prod), Redis 7, Prometheus/Grafana (prod profile)

## Services and ports (local)

| Service | Port | Notes |
|---|---|---|
| nginx edge | **8080** | one origin: `/` → ui, `/api` → backend, `/v1` + `/health` → model-service |
| model-service | 8000 | also direct; `/metrics` for Prometheus |
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

# benchmarks
python bench/latency.py --url http://localhost:8000 --api-key <key> --concurrency 1
python bench/compare_llm.py --data data/decision-v2/test.jsonl --n 50 --dragonfly http://localhost:8000
```

## Repository map

| Path | Contents |
|---|---|
| `model-service/src/dragonfly/schema.py` | `/v1/systemone` request and response shapes; confidence formulas |
| `model-service/src/dragonfly/models/encoder.py` | tier S (ModernBERT + pointer head) |
| `model-service/src/dragonfly/models/decoder.py` | tier M (Qwen3 + LoRA, packing, attention mask) |
| `model-service/src/dragonfly/models/graphs.py` | CUDA graphs for tier S |
| `model-service/src/dragonfly/engine.py` | load/save checkpoints, `Engine.probs`, `Cascade` |
| `model-service/src/dragonfly/batching.py`, `cache.py`, `auth.py` | model worker thread, answer cache, API keys and usage via Redis |
| `model-service/src/dragonfly/api/` | FastAPI app (`app.py`) and entry point (`serve.py`, env config) |
| `model-service/src/dragonfly/plugins/` | plugin API (`api.py`), host, gRPC client (`grpc_plugin.py`, `wire.py`) |
| `model-service/src/dragonfly/training/` | `train.py`, `evaluate.py`, `calibrate.py` |
| `backend/src/` | auth, keys, usage, model proxy, entities, migrations |
| `ui/src/` | pages: Playground, Keys, Usage, Model; `api.ts`, `format.ts` |
| `proto/dragonfly/plugin/v1/plugin.proto` | gRPC plugin protocol |
| `plugins/example-python`, `plugins/example-grpc` | example plugins |
| `bench/`, `scripts/` | benchmarks; `fetch_data.py`, `publish_hf.py` |
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

## Documentation

- [README.md](README.md): overview, quick start, first results
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): services, hot path, model design, cascade, calibration
- [docs/MODEL.md](docs/MODEL.md): tiers, data, training/calibration/eval commands, metrics, results, speed
- [docs/API.md](docs/API.md): `/v1` and `/api` endpoints, request and response shapes, errors, key and usage flow
- [docs/PLUGINS.md](docs/PLUGINS.md): writing in-process and gRPC plugins, closed-source delivery
- [docs/LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md): running with and without Docker, tests, Windows gotchas
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): production server, TLS, deploys with rollback, monitoring, security
- [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md)
