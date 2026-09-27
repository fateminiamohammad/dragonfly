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

### Other model-service endpoints

| Endpoint | Auth | Returns |
|---|---|---|
| `GET /v1/models` | key | model card: tier(s), backbone, trained, temperature, device, CUDA-graph, cache and batch stats, plugins |
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

## How keys and usage flow

```
UI ──create key──► backend ──SADD sha256(key)──► redis set dragonfly:keys
client ──Bearer key──► model-service ──SISMEMBER (cached 30 s)──► redis
model-service ──HINCRBY dragonfly:usage:<sha256>:<day>──► redis ◄──read── backend /api/usage
```

**Rules:**
- Postgres is the source of truth for keys. The backend rebuilds the Redis set on every start.
- Only SHA-256 digests are stored, never plaintext keys.
