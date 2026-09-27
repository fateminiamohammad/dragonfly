# Changelog

All notable changes to this project. The plugin API (`dragonfly.plugins.api.API_VERSION`) follows semver.

## [Unreleased]: v0.1.0

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

### Platform
- Backend (NestJS, Postgres, Redis): accounts, API keys, usage, model info, playground.
- UI (React): playground, API keys, usage, model and plugins.
- Docker compose for local, dev and production; nginx edge with TLS; deploy script with automatic rollback;
  Prometheus/Grafana profile.
- CI for all services; images published to GHCR.
- Benchmarks: latency (`bench/latency.py`) and Dragonfly vs LLM, fair and unfair modes (`bench/compare_llm.py`).
