# Model, training and evaluation

## Tiers

| | Dragonfly-S (speed) | Dragonfly-M (quality) |
|---|---|---|
| Code | `models/encoder.py` | `models/decoder.py` |
| Backbone | ModernBERT-base (150M), fully fine-tuned | Qwen3-1.7B-Base (4B optional), frozen, plus LoRA r=16 |
| Layout | one row per question: `[CLS] instructions - opt1 - opt2 … [SEP] state [SEP]` | one packed sequence per request: `[state][q1][opts…][q2][opts…]` with a custom attention mask |
| Order invariance | no; trained with shuffled options (flip rate 13.4%) | **yes, by construction**: options are isolated and share position IDs |
| Checkpoint | full encoder + head | LoRA adapters + head only; the base comes from the Hub |
| Status | **trained** (`runs/dragonfly-s`) | built and tested; **not trained yet** |

**Shared design:**
- Both use the same **pointer head**: a scaled dot product of the question query against each option key, then a
  softmax over the options.
- Every question type reduces to it:
  - `noul` is a choice between "no" and "yes";
  - `score` is a choice over the ordered levels, and the answer is the expected level.

**Cascade:** with `DRAGONFLY_CHECKPOINT_M` set, tier S answers first and questions below
`DRAGONFLY_CASCADE_THRESHOLD` (default 0.8) are re-asked to tier M.

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

## Results so far

Dragonfly-S, 3 epochs (6 min on an RTX 3090 Ti), temperature 2.07, decision-v2 test (1,440 questions):

| Metric | Overall | choice | noul | score |
|---|---|---|---|---|
| Accuracy | 67.7% | 64.8% | 79.0% | 52.5% |
| ECE | 0.038 | 0.033 | 0.026 | 0.084 |
| auto_rate@5% | 35.6% | 31.8% | 58.5% | 2.1% |

**By source:**
- **Strongest:** IMDB 91.3%, DBpedia 87.9%, AG News 85.3%.
- **Weakest:** MNLI 43.1%, SST-5 47.5%, contrastive 50.0%. These need reasoning; tier M is meant for them.
- **Order flip rate:** 13.4%.

## Speed

**Tier S on GPU:**
- It replays as **CUDA graphs** (`models/graphs.py`).
- An eager forward pass costs about 7 ms of GPU work but about 30–40 ms of CPU time, because it launches about 680
  kernels. A graph replays them in one launch.
- Inputs are padded to (rows, tokens, options) buckets, and each bucket is captured once on first use.
- In fp32 the output matches eager to 1.2e-5.

**Measured with `bench/latency.py`** (unique requests, no cache; 3-question example; RTX 3090 Ti):

| Clients | Round-trip p50 | Model time p50 | Throughput |
|---|---|---|---|
| 1 | 14.1 ms | 8.5 ms | 69 req/s |
| 8 | 58.9 ms | 29.2 ms | 123 req/s |
| 32 | 182.8 ms | 90.5 ms | 153 req/s (460 questions/s) |

**Comparing against an LLM:** `bench/compare_llm.py` measures an LLM in two modes, `reason` (unfair) and
`constrained` (fair), on the same data. It has not been run yet because it needs `LLM_BASE_URL`, `LLM_API_KEY` and
`LLM_MODEL`.

## Publishing a checkpoint

```bash
python scripts/publish_hf.py runs/dragonfly-s <hf-user>/dragonfly-s            # writes the model card from metrics.json
python scripts/publish_hf.py runs/dragonfly-s <hf-user>/dragonfly-s --upload   # uploads to the Hub
```
