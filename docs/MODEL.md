# Model, training and evaluation

## Tiers

| | Dragonfly-S (speed) | Dragonfly-M (quality) |
|---|---|---|
| Code | `models/encoder.py` | `models/decoder.py` |
| Backbone | ModernBERT-base (150M), fully fine-tuned | Qwen3-4B-Base (served; 1.7B also supported), frozen, plus LoRA r=16 |
| Layout | one row per question: `[CLS] instructions - opt1 - opt2 … [SEP] state [SEP]` | one packed sequence per request: `[state][q1][opts…][q2][opts…]` with a custom attention mask |
| Order invariance | no; trained with shuffled options (flip rate 13.4%) | **by construction**: options are isolated and share position IDs (exact in fp32; 0.86% flips in bf16, from rounding on near-ties) |
| Checkpoint | full encoder + head | LoRA adapters + head only; the base comes from the Hub |
| Status | **served:** `runs/dragonfly-s2` (mix-v1, distilled from M-4B, 35 min) | **served:** `runs/dragonfly-m4b` (mix-v1, pointer-head LayerNorm) |

**Shared design:**
- Both use the same **pointer head**: a scaled dot product of the question query against each option key, then a
  softmax over the options.
- Every question type reduces to it:
  - `noul` is a choice between "no" and "yes";
  - `score` is a choice over the ordered levels, and the answer is the expected level.

**Cascade:** with `DRAGONFLY_CHECKPOINT_M` set, tier S answers first and questions below
`DRAGONFLY_CASCADE_THRESHOLD` are re-asked to tier M. The served threshold is **0.70**, chosen on
`data/mix-v1/calibration.jsonl` with `scripts/tune_cascade.py`: the smallest threshold within 1 point of tier M's
accuracy. 0.45 is the measured fast point (below).

**Tier M document cache:** states of 512 tokens or more keep their KV cache (`DRAGONFLY_STATE_CACHE` entries), so
repeated long documents only pay for their questions. It is tested identical to the full pass. Shorter states use the
batched path, because there the cache measured no gain.

## Data

`python scripts/fetch_data.py` downloads **Kev's decision-v2 suite** (pinned commit) into `data/decision-v2/`.

**Splits:**

| Split | Records | Questions |
|---|---|---|
| train | 3,432 | 4,332 |
| calibration | 448 | 568 |
| development | 1,176 | — |
| test | 1,176 | 1,440 |

**Sources:** Banking77, BoolQ, AG News, MNLI, SST-5, Yelp, TREC, DBpedia-14, Amazon, IMDB, plus Kev's contrastive
set. Each dataset keeps its own license.

**Format:** one JSON object per line. Each is a `/v1/systemone` request plus a `label` per question: the criteria
name for `choice`, `true`/`false` for `noul`, the level index for `score`. An optional `target` gives a soft
distribution. Your own data in this format works the same way.

## Commands

All commands run inside `model-service/` (venv) or through the compose `trainer` services.

```bash
# train (writes runs/<out>/: checkpoint + metrics.json)
dragonfly-train --train data/decision-v2/train.jsonl --calibration data/decision-v2/calibration.jsonl \
                --test data/decision-v2/test.jsonl --out runs/dragonfly-s
dragonfly-train --tier M --backbone Qwen/Qwen3-1.7B-Base --grad-checkpointing --batch-size 8 --epochs 2 ... --out runs/dragonfly-m

# re-fit the temperature of a saved checkpoint (no retraining) and re-evaluate
dragonfly-calibrate --checkpoint runs/dragonfly-s --calibration data/decision-v2/calibration.jsonl --test data/decision-v2/test.jsonl

# evaluate any checkpoint
dragonfly-eval --checkpoint runs/dragonfly-s --data data/decision-v2/test.jsonl
```

**In Docker** (recommended on Windows: the container keeps training if your terminal closes):

```bash
docker compose -f docker-compose.local.yml --env-file docker/.env --profile train   up -d trainer     # tier S
docker compose -f docker-compose.local.yml --env-file docker/.env --profile train-m up -d trainer-m   # tier M
docker logs -f dragonfly-trainer
```

**Training details:**
- AdamW with cosine schedule and 6% warmup. Learning rates: 3e-5 for S, 2e-4 for LoRA, 1e-3 for the head.
- bf16 autocast.
- Choice options are shuffled every example.
- Questions that do not fit `--max-length` (default 1,536) are skipped with a warning.

## Metrics (`metrics.json` → `test`)

| Metric | Meaning |
|---|---|
| `accuracy` | argmax equals the label |
| `ece` | expected calibration error of the top probability (10 bins); lower is better |
| `brier`, `nll` | proper scoring rules; lower is better |
| `auto_rate@5%` | share of decisions you can automate, most confident first, while keeping their error ≤ 5% |
| `order_flip_rate` | share of choice questions whose answer changes when the options are shuffled |

Each metric is also reported `by_source` and `by_type`.

## Results

### v0.2 (served): S2 + M-4B, trained on mix-v1

All numbers on one RTX 3090 Ti. Test splits never overlap training: `scripts/build_mix.py` drops any record whose
document appears in a test or calibration split.

| decision-v2 test (1,440 questions) | Tier S2 | Tier M-4B | **Cascade 0.70** (served) | Cascade 0.45 (fast) |
|---|---|---|---|---|
| Accuracy | 80.1% | 86.2% | **84.4%** | 82.3% (measured by `compare_systemone`) |
| ECE | 0.063 | 0.047 | - | - |
| auto_rate@5% | 61.6% | 77% | - | - |
| Option-order flip rate | 4.5% | 0.14% | - | - |
| Questions answered by tier M | - | 100% | 26.2% | ~20% |

- **By source (S2):** contrastive 94.4%, DBpedia 90.5%, IMDB 90.0%, AG News 89.0%, TREC 88.8%, MNLI 73.3%,
  Yelp 71.9%, Banking77 71.6%, BoolQ 61.3%, Amazon 56.3%, SST-5 56.3%. Score questions (5-level ratings) remain the
  weakest: 56.7%.
- **typed-decisions test (1,950 decisions):** M-4B 79.3%; served cascade 79.2%.
- **Training:** M-4B with LoRA r=16, a LayerNorm in the pointer head (`head_norm`; without it, 4B training diverged),
  LoRA lr 1e-4, head lr 3e-4. S2: ModernBERT-base on M-4B's calibrated probabilities mixed 50/50 with the labels,
  3 epochs, 35 min.
- **From v0.1 to v0.2:** tier S 68.2% → 80.1%, tier M 76.9% → 86.2%, served cascade 76.6% → 84.4%.

### v0.1: S + M-1.7B, trained on decision-v2 only


All numbers are on the decision-v2 test split (1,440 questions), RTX 3090 Ti.

| | Dragonfly-S | Dragonfly-M | **Cascade S→M (0.55)** |
|---|---|---|---|
| Accuracy | 67.7% | 76.9% | **77.5%** |
| ECE (lower is better) | 0.038 | 0.027 | - |
| auto_rate@5% | 35.6% | 54.2% | - |
| Option-order flip rate | 13.4% | 0.86% | - |
| Questions answered by tier M | - | 100% | 42.3% |
| Training time | 6 min | 17 min | - |
| Temperature (fitted on calibration) | 2.07 | 1.63 | - |

**Distilled tier S (served):** tier S trained on tier M's calibrated probabilities mixed 50/50 with the labels
(`scripts/distill_targets.py`). It improves every metric:

| | Dragonfly-S | Dragonfly-S distilled |
|---|---|---|
| Accuracy | 67.7% | **68.2%** |
| ECE | 0.038 | **0.032** |
| Brier | 0.394 | **0.388** |
| auto_rate@5% | 35.6% | **39.4%** |

**Cascade operating points** (distilled S + M; threshold chosen on the calibration split, then measured once on test):

| Threshold | Test accuracy | Escalated to tier M | Use |
|---|---|---|---|
| 0.20 | 72.2% | 14.2% | fastest |
| **0.45** (default) | **76.6%** | **35.5%** | balanced: M's accuracy, with 64.5% of questions staying on the fast tier |
| 0.55 with the original S | 77.5% | 42.3% | earlier default |

**By type (M):** choice 80.0%, noul 85.7%, score 49.6%. Score questions (5-level ratings such as SST-5 and Amazon) are
the weakest for both tiers.

**By source (M):** DBpedia 93.1%, IMDB 93.8%, contrastive 87.8%, AG News 86.7%, TREC 84.5%, BoolQ 77.5%, Yelp 71.3%,
MNLI 64.7%, Banking77 58.6%, Amazon 53.8%, SST-5 41.3%.

**For reference:** Kev reports 85.6% in-distribution for its 4B model. Dragonfly-M is a 1.7B model trained for
17 minutes.

## Speed

**Both tiers replay as CUDA graphs** (`models/graphs.py`). An eager forward pass is dominated by kernel launches:
- tier S: about 680 launches, about 7 ms of GPU work but about 30–40 ms wall;
- tier M: about 3,100 launches, 36 ms of GPU work but about 115 ms wall.

A graph replays the whole pass in one launch:
- Inputs are padded to shape buckets, captured once each on first use.
- Padding is masked, and the results match eager in fp32 (tested).
- Tier M uses fine token buckets (64, 96, 128, 160, …), because padding a decoder is real GPU work.

| Pass | Eager | CUDA graph |
|---|---|---|
| Tier S, 3-question request | 30 ms | **8.5 ms** |
| Tier M, 1 request (136 tokens) | 113.5 ms | **20.6 ms** (LoRA merged into the weights at load: a further about 4 ms) |
| Tier M, 8 requests batched | 113.5 ms | 102 ms |

**End to end with `bench/latency.py`** (unique requests, no answer cache; 3-question example request):

| Setup | Clients | Round-trip p50 | Model p50 | Throughput |
|---|---|---|---|---|
| Tier S only | 1 | **14.1 ms** | 8.5 ms | 69 req/s |
| Tier S only | 32 | 182.8 ms | 90.5 ms | 153 req/s (460 questions/s) |
| Cascade S→M (2 of the 3 questions escalate), LoRA merged | 1 | **31.0 ms** | 24.4 ms | 32 req/s |
| Cascade S→M, LoRA merged | 32 | 618 ms | 376 ms | 45 req/s (136 questions/s; tier M compute-bound at about 50% of bf16 peak) |

**v0.2 (S2 → M-4B, threshold 0.70), same benchmark:**

| Clients | Round-trip p50 | Model p50 | Throughput |
|---|---|---|---|
| 1 | **49.3 ms** | 42.5 ms | 20 req/s |
| 8 | 304 ms | 232 ms | 26 req/s |
| 32 | 991 ms | 495 ms | 31 req/s (93 questions/s) |

The example request sends 2 of its 3 questions to the 4B tier, which is compute-bound under load (one M-4B request:
21 ms merged). Tier S alone answers in about 5 ms. More replicas or GPUs scale this out (DEPLOYMENT.md).

**Idle GPU clocks.** A desktop GPU drops to its idle clock (210 MHz on the RTX 3090 Ti, P8) within seconds of the last
request, and the first request after a pause then takes 200–350 ms instead of 20–40 ms. The next one is fast again.
Small keep-alive kernels did not hold the clock up (measured). For steady latency on a server, lock the clocks:
`sudo nvidia-smi -lgc 1500,2100` on Linux, or "Prefer maximum performance" in the NVIDIA Control Panel on Windows.
Benchmarks here run continuously, so they measure the ramped-up GPU.

**Serving optimizations for tier M:**
- LoRA is merged into the base weights at load (`Engine.load(..., merge=True)`), which is tested identical.
- Common graph shapes are captured at startup (`DRAGONFLY_WARMUP`), so no request pays the one-off capture. p95 went
  from 365 to 38 ms.
- Buckets go up to 64 rows, so large escalated batches never fall back to eager.

**Not adopted: ONNX Runtime / TensorRT export for tier S.**
- CUDA graphs already remove the launch overhead: 8.5 ms model time against about 7 ms of pure GPU work.
- ONNX Runtime GPU measured slower than the torch path on this machine (OCR: 535 vs 144 ms).
- We keep it as a possible follow-up for TensorRT FP8/INT8, not as a default.

## Versus a common LLM

All runs use the same RTX 3090 Ti, the same decision-v2 test requests, and `bench/compare_llm.py`. The LLMs run
locally in Ollama, and Dragonfly bypasses its answer cache (`Cache-Control: no-cache`). Dragonfly is the served
configuration: distilled tier S plus tier M, cascade threshold 0.45.

| Against | Requests | Dragonfly accuracy / p50 | LLM accuracy / p50 (p95) | Dragonfly is |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct, answer only (fair) | 100 | **79.1% / 18 ms** | 74.4% / 103 ms (208 ms) | **5.7× faster** |
| Qwen2.5-7B-Instruct, reasoning first | 100 | 79.1% / 18 ms | 76.7% / 995 ms (1.6 s) | **55× faster** |
| Qwen3-8B, thinking model | 50 | **84.6% / 16 ms** | 80.0% / 1,658 ms (7.0 s) | **105× faster** (p50 and mean) |

**What this shows:**
- Dragonfly is **more accurate than every LLM we tested**.
- It is **55–105× faster than LLMs that reason or think** on the same GPU.
- It is **about 6× faster than a small local LLM that only outputs an answer**. That LLM is fast because it's local and
  its output is tiny.

**About the 200× target:** we have **not measured 200× against a local LLM**.
- Hosted APIs add network and queueing time. Jev's 20–200× was measured against hosted frontier models that took
  3–329 s. A hosted comparison is one command (`LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`), but it needs a key.
- To reach 200× against the local thinking model, Dragonfly would need about 8 ms end to end. Tier S alone is at
  14 ms today.

Raw results: `runs/compare-llm-*.json`.

## Versus Kev, Laya and Jev

**Setup:**
- Everything we measured ran on one RTX 3090 Ti with `bench/compare_systemone.py`.
- Every system received byte-identical `/v1/systemone` requests and was scored against the same labels, one client,
  after warm-up.
- Kev and Laya were served exactly as their READMEs say, from pinned versions (`bench/kev/`, `bench/laya/`; compose
  profiles `kev`, `laya`).
- Jev is closed. Its rows are **published** numbers from the typed-decisions leaderboard, measured by others over the
  network. They are not our measurement.

### Benchmark 1: typed-decisions

400 cases and 2,000 decisions from four workflows (agent-trace observability, customer service, invoice processing,
security incidents); 390 requests and 1,950 decisions after the harness drops cases it cannot send. It's the public
benchmark where Jev and Laya publish their numbers. Measured 2026-09-28 (`runs/v02-td-*.json`).

| System | Trained on this benchmark's train split? | Accuracy | p50 | p95 |
|---|---|---|---|---|
| **Dragonfly** (general, S2 → M-4B) | yes (part of mix-v1) | **79.2%** | 125 ms | 178 ms |
| Dragonfly, fast threshold 0.45 | yes | 77.6% | 107 ms | 137 ms |
| Laya-typed-decisions (421M) | yes | 76.8% | 57 ms | 79 ms |
| Dragonfly-td specialist (tier S, 150M) | yes | 72.8% | **51 ms** | 74 ms |
| Jev 1.13.0 | no | 72.7% | 710 ms (hosted) | - |
| Kev-4B | no | 67.0% | 114 ms | 151 ms |
| Kev-0.8B | no | 46.3% | 32 ms | 94 ms |
| Laya (English base) | no | 36.3% | 59 ms | 80 ms |
| Prior (ignores the input) | - | 47.0% | - | - |

**Reading the typed-decisions results:**
- Dragonfly's general model is the **most accurate** system measured: +2.4 points over Laya-typed-decisions and
  +6.5 over Jev's published score. Jev is scored zero-shot there, so that comparison is not like-for-like.
- It is **not the fastest** on this benchmark. Requests carry long documents and five questions, S2 is unsure on 71%
  of them, and those go to the 4B tier. Laya-td is 2.2× faster. For speed, the tier S specialist answers in 51 ms
  (72.8%).
- Laya reproduces its published numbers here (76.8% vs its card's 76.6%; 36.3% vs 36.2%), a good sign that the setup
  is fair.

### Benchmark 2: decision-v2

All 1,166 test requests (1,430 questions) from Kev's suite. Dragonfly trained on its train split, and Kev on the same
sources. Kev never saw these test questions (checked against every file in its repo). One client, each system alone
on the GPU (`runs/v02-dv2-*.json`).

| System | Accuracy | p50 | p95 | Throughput (1 client) |
|---|---|---|---|---|
| Kev-4B | **87.9%** | 43.0 ms | 225 ms | 14.1 req/s |
| Kev-0.8B | 85.0% | 16.7 ms | 163 ms | 25.8 req/s |
| **Dragonfly** (threshold 0.70, default) | 84.6% | 16.5 ms | 73 ms | 36.1 req/s |
| **Dragonfly** (threshold 0.45, fast) | 82.3% | **12.3 ms** | **46 ms** | **48.2 req/s** |
| Laya (English base, zero-shot) | 64.4% | 52.2 ms | 72 ms | 12.5 req/s |

v0.1 measured Kev-0.8B at 43 ms p50 on a 200-request sample while sharing the GPU; alone on the GPU it is as fast as
Dragonfly at the median.

### Summary

- **decision-v2:** Kev-4B is the most accurate (+3.3 points over Dragonfly; the gap was 12.1 points in v0.1).
  Dragonfly is 2.6× faster than Kev-4B at the median and has the best tail latency and throughput of every system.
  Kev-0.8B is a close match: 0.4 points more accurate, the same median, a 2.2× worse p95.
- **typed-decisions:** Dragonfly is the most accurate system measured, and slower than Laya.
- **Jev:** against its *published* hosted latencies (236–710 ms), Dragonfly is 5–30× faster, and more accurate on
  typed-decisions.

### Jev's published benchmarks

Jev is closed and needs a paid, waitlisted key, so we list what others have **published**, with their conditions:

| Source | What was measured | Jev result |
|---|---|---|
| [typed-decisions leaderboard](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | 400 cases, zero-shot | 72.7% accuracy, Brier 0.148, **710 ms** p50 end to end |
| [Laya model card](https://huggingface.co/convaiinnovations/laya) | typed-decisions and Banking77 | 72.7% typed-decisions, 87.0% Banking77; **236–276 ms** p50 per question; ECE 0.246 |
| [Kev README / model cards](https://github.com/jaredpalmer/kev) | Kev's suites | 85.7% (Kev-9B: 85.2%); automates 70% of decisions at 5% error (Kev: 45–57%); confidently wrong on 3.7% |
| [Opper](https://opper.ai/blog/jev-vs-kev-open-decision-model) | 362 fresh arXiv / Stack Exchange / GitHub questions | about Kev-4B's accuracy (within 2 points); **about 275 ms** (Kev-4B: about 220 ms) |
| [TypeSafe](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | vendor claim | **70–500 ms**, "20–200× faster" than LLMs writing full answers |
| [AIMultiple](https://aimultiple.com/decision-models) | 50 browser-agent tasks | 17/50 solved, 4.2 s per attempt (GPT-6 Astra: 47/50) |
| [Jev 101](https://jev101.org/jev-alternatives) | independent 49-task classifier benchmark | 96.6% (Laya: 58.3%, Von: 70.4%) |

The harness measures Jev directly once you have a key:

```bash
TYPESAFE_API_KEY=... python bench/compare_systemone.py --data data/typed-decisions/test.jsonl --n 400 \
  --system dragonfly=http://127.0.0.1:8000@DRAGONFLY_API_KEY --system jev=https://api.typesafe.ai@TYPESAFE_API_KEY
```

## Publishing a checkpoint

```bash
python scripts/publish_hf.py runs/dragonfly-s <hf-user>/dragonfly-s            # writes the model card from metrics.json
python scripts/publish_hf.py runs/dragonfly-s <hf-user>/dragonfly-s --upload   # uploads to the Hub
```
