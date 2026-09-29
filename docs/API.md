# API

Two HTTP APIs. **Clients use `/v1`** (the model-service). **The UI uses `/api`** (the backend). Both are behind nginx on
one origin: `http://localhost:8080` locally, `https://<APP_DOMAIN>` in production.

## Model-service: `/v1` (for applications)

TypeSafe-compatible: the TypeSafe and Kev SDKs work by pointing their base URL at Dragonfly. Every `/v1` call needs
`Authorization: Bearer <key>`. Keys come from the UI, or from `DRAGONFLY_API_KEYS` for static ones.

### `POST /v1/systemone` (alias: `POST /v1/decide`)

**Request:**

```json
{
  "state": "text, or any JSON object/array the answer depends on",
  "model": "dragonfly-latest",
  "questions": {
    "intent":   { "type": "choice", "instructions": "Which intent?", "criteria": { "card_arrival": "Card has not arrived", "refund": null } },
    "escalate": { "type": "noul",   "instructions": "Escalate to a human?", "criteria": { "true": "optional description", "false": null } },
    "mood":     { "type": "score",  "instructions": "How does the customer feel?", "criteria": ["negative", "neutral", "positive"] }
  }
}
```

| Question type | `criteria` | Answer fields |
|---|---|---|
| `choice` | object: option name → description (or `null`); 1–255 options | `choice`, `probabilities` (by name), `confidence` |
| `noul` | optional object with `"true"` / `"false"` descriptions | `noul` (p(yes)), `confidence` |
| `score` | ordered list of level descriptions; 1–255 levels | `score` (expected level), `probabilities` (by level index), `legend`, `confidence` |

**Response:**

```json
{
  "model": "dragonfly-latest",
  "answers": {
    "intent":   { "type": "choice", "choice": "card_arrival", "confidence": 0.62, "probabilities": { "card_arrival": 0.81, "refund": 0.19 }, "tier": "S" },
    "escalate": { "type": "noul", "noul": 0.21, "confidence": 0.58, "tier": "S" },
    "mood":     { "type": "score", "score": 0.4, "legend": { "0": "negative", "1": "neutral", "2": "positive" }, "probabilities": { "0": 0.62, "1": 0.36, "2": 0.02 }, "confidence": 0.51, "tier": "M" }
  },
  "usage": { "input_tokens": 312, "output_tokens": 0 },
  "latency_ms": 8.5,
  "cached": false
}
```

**Response fields:**
- `confidence` uses TypeSafe's formulas: 0 at uniform, 1 at certainty.
- `tier` is the model that answered (`S`, or `M` via the cascade).
- `latency_ms` is the model time of the batch the request ran in; it is 0 for a cache hit.
- Plugins may add fields, e.g. `review_required` from the example policy plugin.
- Every response carries `x-typesafe-request-id`, `x-request-id` and `server-timing` headers.

**Errors:**

| Status | When |
|---|---|
| 401 | missing or invalid API key |
| 422 | invalid request, or a question whose options don't fit the model's token limit |
| 4xx from a plugin | a plugin rejected the request, e.g. 413 for an oversized state |
| 500 | a fail-closed plugin errored |

### Risk-controlled answers (`max_error`)

Calibrated confidence says how sure the model is; `max_error` turns it into a promise. Add it to any request
(0 < max_error < 0.5) on a server with a fitted risk table (`DRAGONFLY_RISK`, made by `scripts/fit_risk.py`):

```json
{"state": "...", "max_error": 0.05, "questions": {"intent": {"type": "choice", "criteria": {"refund": null, "billing": null, "other": null}}}}
```

Every answer then gains:

| Field | Meaning |
|---|---|
| `decided` | `true`: on held-out data, answers at this confidence were wrong at most `max_error` of the time (95% Clopper-Pearson bound, Bonferroni-corrected over the candidate thresholds). Act on it automatically. |
| `set` | the smallest set of option keys that contains the right answer with probability ≥ 1 − `max_error` (split conformal prediction), most likely first. One key = sure; several = send just these to a person or an LLM. |
| `risk` | the target actually used (the largest fitted target ≤ `max_error`: 0.005, 0.01, 0.02, 0.05, 0.1, 0.2), the confidence threshold and the calibration size. |

The promise holds for data like the calibration data; `fit_risk.py` reports the realized error on a test split next
to each target. A request without `max_error` is unchanged. Answers from a specialist are never `decided` (the table
was fitted on the general model). Without a table, `max_error` returns 400.

### Batch / map-reduce (`/v1/batch`)

Many decisions in one call, e.g. every row of a table:

```bash
curl -s localhost:8000/v1/batch -H 'content-type: application/json' -d '{"requests": [
  {"state": "ticket 1 ...", "questions": {"urgent": {"type": "noul", "instructions": "Urgent?"}}},
  {"state": "ticket 2 ...", "questions": {"urgent": {"type": "noul", "instructions": "Urgent?"}}}]}'
# -> {"results": [<a /v1/systemone response per request, in order>], "stats": {"decisions_per_s": ...}}
```

- Each request goes through the normal pipeline (plugins, cache, usage), shortest first so batches pad less, at
  **batch priority**: live `/v1/systemone` requests always get the next forward pass, and batch work fills the rest.
- A bad request yields `{"error": ...}` in its place; the rest of the batch is answered.
- Up to `DRAGONFLY_BATCH_MAX` requests (default 10,000); `DRAGONFLY_BATCH_CONCURRENCY` in flight (default 512).
- `POST /v1/batch/jobs` does the same in the background and returns `{"id"}`; `GET /v1/batch/jobs/{id}` shows
  `status`, `done`/`total` and, when done, `results` and `stats` (`?results=false` for progress only). Jobs live in
  the replica that started them (the last 20 are kept). The UI **Batch** page runs a question over a CSV this way.

### Specialists (the swarm)

With `DRAGONFLY_SPECIALISTS` set, `model` selects a specialist by name, or `"auto"` to let the general model route.
`dragonfly-latest` (and the SDK defaults `jev-latest`, `kev-latest`) is the general model. The response then carries
`"specialist": "<name>" | "general"`, and an unknown model name is a 422. See [SWARM](SWARM.md).

### Images and audio in the state

Any object `{"type": "image" | "audio", "data": "<base64>"}` inside `state` is converted to text before the decision:
- images become OCR `text` plus the tags they show (`shows`);
- audio becomes a `transcript`.

Optional keys: `tasks` (`["ocr", "tags"]`), `ocr_lang` (`en` / `arabic`), `language` (`en` / `fa`). The response adds
`usage.media_ms`. Details: [PERCEPTION.md](PERCEPTION.md).

### `POST /v1/perceive`

Converts up to 16 media items to text without deciding. Request: `{"items": [{"id", "type", "data", ...}]}`. The
response gives `items[id]` with `ocr` / `tags` / `transcript`, plus `latency_ms` per stage.

Status codes:
- 404 if the server has no perception-service (`PERCEPTION_URL` unset);
- 422 for bad media;
- 503 if the perception-service is unreachable.

### Other model-service endpoints

| Endpoint | Auth | Returns |
|---|---|---|
| `GET /v1/models` | key | model card: tier(s), backbone, trained, temperature, device, CUDA-graph, cache and batch stats, plugins |
| `POST /v1/flows/run` `{flow, state, mermaid?}` | key | run an inline agent flow: path, every step's answers and latency ([SWARM](SWARM.md#agent-flows-chains-of-dragonflies)) |
| `POST /v1/flows/:name/run` `{state, mermaid?}` | key | run a saved flow |
| `GET /v1/specialists` | key | the swarm's specialists: name, description, tier, metrics, loaded, requests served ([SWARM](SWARM.md)) |
| `GET /health` | none | `{"status": "ok", "version", "trained"}` |
| `GET /metrics` | none (internal network only) | Prometheus metrics |

## Backend: `/api` (for the UI)

Every route except `/api/health` and `/api/auth/login` needs `Authorization: Bearer <JWT>` from login.

| Method and path | Role | Purpose |
|---|---|---|
| `GET /api/health` | public | liveness |
| `POST /api/auth/login` `{email, password}` | public, 10/min | returns `{token, user}` |
| `GET /api/me` | any | current user |
| `POST /api/users` `{email, password (≥12), role}` | admin | create a user |
| `GET /api/keys` | any | your API keys; the plaintext is never returned again |
| `POST /api/keys` `{name}` | any | create a key; the response includes `key` **once** |
| `DELETE /api/keys/:id` | any | revoke; the model-service stops accepting it within 30 s |
| `GET /api/usage?days=30` | any | requests, questions and tokens per key per UTC day (1–90 days) |
| `GET /api/model` | any | proxies `/v1/models` |
| `POST /api/model/playground` | any, 120/min | proxies `/v1/systemone` with the internal key |
| `POST /api/model/perceive` | any, 60/min | proxies `/v1/perceive` (images/audio → text); JSON bodies up to 30 MB |
| `GET /api/review?limit=50` | any | the human-review queue: pending and reviewed counts, items with their answer options |
| `POST /api/review/:id` `{label}` | any | label an item (moves it to the reviewed list) |
| `POST /api/review/:id/skip` | any | drop an item without labelling it |
| `GET /api/review/export` | any | reviewed items as training JSONL |
| `GET /api/specialists` | any | the swarm's specialists (from `/v1/specialists`) and the 20 latest training jobs with progress |
| `POST /api/specialists` `{name, description, tier, format: csv\|jsonl, data, question?, epochs?}` | any, 10/min | validate an upload and queue a training job ([SWARM](SWARM.md#train-your-own-ui--specialist-in-minutes)) |
| `DELETE /api/specialists/:name` | admin | remove a specialist (every replica reloads) |
| `POST /api/batch/jobs` `{requests}` / `GET /api/batch/jobs/:id` | any, 10/min | batch decisions from the UI (proxies `/v1/batch/jobs`) |
| `GET /api/flows` | any | saved flows |
| `PUT /api/flows/:name` `{flow}` | any | save a flow (structure checked; stored in Redis for every replica) |
| `DELETE /api/flows/:name` | any | delete a flow |
| `POST /api/flows/try` `{flow, state}` / `POST /api/flows/:name/run` `{state}` | any, 60/min | run a flow from the UI (with its Mermaid diagram) |

## How keys and usage flow

```
UI ──create key──► backend ──SADD sha256(key)──► redis set dragonfly:keys
client ──Bearer key──► model-service ──SISMEMBER (cached 30 s)──► redis
model-service ──HINCRBY dragonfly:usage:<sha256>:<day>──► redis ◄──read── backend /api/usage
```

**Rules:**
- Postgres is the source of truth for keys. The backend rebuilds the Redis set on every start.
- Only SHA-256 digests are stored, never plaintext keys.
