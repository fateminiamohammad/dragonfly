"""HTTP API. TypeSafe-compatible (POST /v1/systemone, GET /v1/models, x-typesafe-request-id, bearer auth), so the
official TypeSafe and Kev clients work by changing base_url. POST /v1/decide is the same endpoint under Dragonfly's
own name. Each answer also carries `tier` (which model answered it); clients that don't know the field ignore it.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .. import __version__
from ..auth import KeyStore
from ..batching import Worker
from ..cache import AnswerCache, record_key
from ..media import MediaError, MediaResolver, find_media
from ..plugins import PluginError, PluginHost
from ..schema import MODEL_NAMES, DecideRequest, to_answers, to_record

REQUESTS = Counter("dragonfly_requests_total", "Decision requests", ["status"])
QUESTIONS = Counter("dragonfly_questions_total", "Questions answered", ["type", "tier"])
CACHE = Counter("dragonfly_cache_total", "Answer cache lookups", ["result"])
LATENCY = Histogram("dragonfly_request_seconds", "Time inside the service per decision request",
                    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
MODEL_LATENCY = Histogram("dragonfly_model_seconds", "Forward-pass time of the batch a request ran in",
                          buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1))


def create_app(worker: Worker, plugins: PluginHost | None = None, api_keys: list[str] | None = None,
               keys: KeyStore | None = None, cache: AnswerCache | None = None,
               media: MediaResolver | None = None) -> FastAPI:
    """No keys configured = open server (local default). `media` enables images/audio in the state (perception-service)."""
    plugins = plugins or PluginHost([])
    keys = keys or KeyStore(api_keys)
    cache = cache or AnswerCache(0)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        plugins.shutdown()
        worker.close()
        if media is not None:
            await media.close()

    app = FastAPI(title="dragonfly", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def auth_and_ids(request: Request, call_next):
        started = time.perf_counter()
        request.state.key_digest = None
        if keys.enabled and request.url.path.startswith("/v1"):
            given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            request.state.key_digest = await keys.check(given)
            if request.state.key_digest is None:
                resp = JSONResponse({"detail": "missing or invalid API key; send Authorization: Bearer <key>"}, 401,
                                    {"www-authenticate": "Bearer"})
                resp.headers["x-typesafe-request-id"] = uuid.uuid4().hex
                return resp
        resp = await call_next(request)
        rid = request.headers.get("x-typesafe-request-id") or request.headers.get("x-request-id") or uuid.uuid4().hex
        resp.headers["x-typesafe-request-id"] = rid
        resp.headers["x-request-id"] = rid
        resp.headers["server-timing"] = f"app;dur={(time.perf_counter() - started) * 1000:.2f}"
        return resp

    async def decide(req: DecideRequest, request: Request, background: BackgroundTasks) -> dict:
        started = time.perf_counter()
        media_ms = 0.0
        try:
            if find_media(req.state):
                if media is None:
                    raise MediaError("this server has no perception-service (PERCEPTION_URL): images and audio are not supported")
                state, media_ms = await media.resolve(req.state)
                req = req.model_copy(update={"state": state})
            req = await plugins.aon_request(req)
            record, meta = to_record(req)
            key = record_key(record, req.model)
            # "Cache-Control: no-cache" skips the lookup (standard HTTP semantics); the answer is still stored
            hit = None if "no-cache" in request.headers.get("cache-control", "") else cache.get(key)
            CACHE.labels("hit" if hit else "miss").inc()
            if hit:
                probs, stats = hit
                latency = 0.0
            else:
                probs, stats = await asyncio.wrap_future(worker.submit(record))
                cache.put(key, (probs, stats))
                latency = stats["latency_ms"]
                MODEL_LATENCY.observe(latency / 1000)
            answers = to_answers(probs, meta)
            for m, tier in zip(meta, stats["tiers"]):
                answers[m["id"]]["tier"] = tier
            body = {
                "model": req.model,
                "answers": answers,
                "usage": {"input_tokens": stats["tokens"], "output_tokens": 0, "media_ms": media_ms},
                "latency_ms": latency,
                "cached": bool(hit),
            }
            body = await plugins.aon_decision(req, body)
        except PluginError as e:
            REQUESTS.labels("rejected").inc()
            raise HTTPException(e.status, str(e)) from e
        except MediaError as e:
            REQUESTS.labels("media_error").inc()
            raise HTTPException(e.status, str(e)) from e
        except ValueError as e:
            REQUESTS.labels("invalid").inc()
            raise HTTPException(422, str(e)) from e
        REQUESTS.labels("ok").inc()
        for m, tier in zip(meta, stats["tiers"]):
            QUESTIONS.labels(m["type"], tier).inc()
        LATENCY.observe(time.perf_counter() - started)
        background.add_task(keys.record_usage, request.state.key_digest, len(meta), stats["tokens"])
        return body

    app.post("/v1/systemone")(decide)
    app.post("/v1/decide")(decide)

    @app.post("/v1/perceive")
    async def perceive(payload: dict):
        """Images/audio -> text only (no decision): proxied to the perception-service, with the same API-key auth."""
        if media is None:
            raise HTTPException(404, "this server has no perception-service (PERCEPTION_URL)")
        try:
            return await media.perceive(payload)
        except MediaError as e:
            raise HTTPException(e.status, str(e)) from e

    @app.get("/v1/models")
    def models():
        engine = worker.engine.describe()
        card = {
            "description": "Dragonfly System One decision model",
            **engine,
            "plugins": plugins.describe(),
            "cache": cache.stats(),
            "batches": {"count": worker.batches, "requests": worker.requests, "queued": worker.queue.qsize()},
        }
        return {"models": [{"name": n, **card} for n in MODEL_NAMES]}

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "trained": worker.engine.describe()["trained"]}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
