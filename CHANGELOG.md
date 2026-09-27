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
