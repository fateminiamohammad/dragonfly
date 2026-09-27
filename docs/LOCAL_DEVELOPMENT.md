# Local development

## With Docker (whole stack)

```bash
cp docker/.env.example docker/.env      # edit passwords/keys; never commit it (all env files are gitignored)
docker compose -f docker-compose.local.yml --env-file docker/.env up -d --build
```

| URL | What |
|---|---|
| http://localhost:8080 | UI; log in with `ADMIN_EMAIL` / `ADMIN_PASSWORD` from `docker/.env` |
| http://localhost:8080/v1/systemone | model API (Bearer key) |
| http://localhost:8080/api/... | backend API (Bearer JWT) |
| http://localhost:8080/health | model-service health |
| http://localhost:8000 | model-service directly |
| perception-service:8001 | internal only (reach it through `/v1/perceive`); first start downloads about 3 GB of models |
| localhost:5434 | Postgres (user and password from `docker/.env`) |

**Compose profiles:**

| Profile | Service | Purpose |
|---|---|---|
| `train` | `trainer` | train Dragonfly-S |
| `train-m` | `trainer-m` | train Dragonfly-M |
| `plugins` | `plugin-policy` | example gRPC plugin |
| `bench` | `bench` | Dragonfly vs an LLM |
| `llm` | `llm` | local Ollama LLM for the comparison (`ollama pull qwen2.5:7b-instruct`, `qwen2.5vl:7b`) |

```bash
docker compose -f docker-compose.local.yml --env-file docker/.env ps           # status
docker logs -f dragonfly-model-service                                          # logs
docker compose -f docker-compose.local.yml --env-file docker/.env down          # stop (volumes kept)
```

**Settings in `docker/.env` that matter locally:**
- `DRAGONFLY_CHECKPOINT=/runs/dragonfly-s`: the trained model. `base:answerdotai/ModernBERT-base` runs an untrained
  head instead.
- `TORCH_INDEX_URL` and `DRAGONFLY_DEVICE`: the cu126 wheel plus GPU is the default. For a CPU-only machine, use the
  `whl/cpu` wheel with `DRAGONFLY_DEVICE=cpu`.
- `DRAGONFLY_API_KEYS` must contain `MODEL_SERVICE_API_KEY`, or the UI playground gets 401.
- `DRAGONFLY_GRPC_PLUGINS=policy@plugin-policy:50051` needs the `plugins` profile running. The model-service refuses
  to start without a configured plugin (fail closed).

## Without Docker

**model-service:**

```bash
cd model-service
python -m venv .venv && . .venv/Scripts/activate      # Linux/macOS: .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or whl/cpu
pip install -e ".[train,grpc,dev]"
pytest                                    # 64 tests, tiny random models, no downloads; the 2 CUDA-graph tests need a GPU
ruff check --config pyproject.toml src tests ../plugins ../bench ../scripts
DRAGONFLY_CHECKPOINT=../runs/dragonfly-s dragonfly-serve     # :8000
```

**perception-service:**

```bash
cd perception-service
python -m venv .venv && . .venv/Scripts/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"
pytest                                  # 14 tests, stub models, no downloads
PERCEPTION_PORT=8001 dragonfly-perception
```

**backend** (needs Postgres and Redis; start them with compose: `up -d postgres redis`):

```bash
cd backend && npm ci
npm run typecheck && npm run lint && npm test && npm run build
DB_PORT=5434 POSTGRES_PASSWORD=<from docker/.env> JWT_SECRET=dev ADMIN_EMAIL=a@b.c ADMIN_PASSWORD=change-me-please npm run start:dev   # :3000
```

**ui:**

```bash
cd ui && npm ci
npm test && npm run build
npm run dev          # :5173, proxies /api to localhost:3000
```

## Windows gotchas

- **Git Bash rewrites container paths.** `/runs/x` becomes `C:/Program Files/Git/runs/x`. Prefix docker commands that
  pass container paths with `MSYS_NO_PATHCONV=1`.
- **File-watch reload over bind mounts is unreliable.** After editing `backend/src` or `model-service/src`, run
  `docker restart dragonfly-backend` (or `dragonfly-model-service`).
- **Long GPU jobs belong in Docker.** Training started directly from a Claude Code session died whenever the session
  restarted. Run it through the `trainer` compose services.

## Checks CI runs (keep them green)

| Workflow | Checks |
|---|---|
| `model-service-ci.yml` | ruff and pytest on CPU PyTorch; runtime image build |
| `perception-service-ci.yml` | ruff and pytest on CPU PyTorch; runtime image build |
| `web-ci.yml` | backend typecheck, lint, test and build; ui test and build |
| `publish.yml` | on `main`, pushes `ghcr.io/fateminiamohammad/dragonfly-{model-service,perception-service,backend,ui,plugin-policy}:<sha>` and `:latest` |
