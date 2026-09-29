"""HTTP API. TypeSafe-compatible (POST /v1/systemone, GET /v1/models, x-typesafe-request-id, bearer auth), so the
official TypeSafe and Kev clients work by changing base_url. POST /v1/decide is the same endpoint under Dragonfly's
own name. Each answer also carries `tier` (which model answered it); clients that don't know the field ignore it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import ValidationError

from .. import __version__, twostage
from ..auth import KeyStore
from ..batching import Worker
from ..cache import AnswerCache, record_key
from ..engine import Cascade
from ..flows import FlowError, mermaid, run_flow
from ..media import MediaError, MediaResolver, find_media
from ..plugins import PluginError, PluginHost
from ..risk import RiskTable
from ..schema import MAX_OPTIONS, MODEL_NAMES, DecideRequest, question_confidence, render, to_answers, to_record

REQUESTS = Counter("dragonfly_requests_total", "Decision requests", ["status"])
QUESTIONS = Counter("dragonfly_questions_total", "Questions answered", ["type", "tier"])
CACHE = Counter("dragonfly_cache_total", "Answer cache lookups", ["result"])
LATENCY = Histogram("dragonfly_request_seconds", "Time inside the service per decision request",
                    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
MODEL_LATENCY = Histogram("dragonfly_model_seconds", "Forward-pass time of the batch a request ran in",
                          buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1))


def batch_stats(results: list[dict], seconds: float) -> dict:
    ok = [r for r in results if "answers" in r]
    questions = sum(len(r["answers"]) for r in ok)
    return {"requests": len(results), "errors": len(results) - len(ok), "questions": questions,
            "seconds": round(seconds, 3), "decisions_per_s": round(questions / seconds, 1) if seconds else None}


def create_app(worker: Worker, plugins: PluginHost | None = None, api_keys: list[str] | None = None,
               keys: KeyStore | None = None, cache: AnswerCache | None = None,
               media: MediaResolver | None = None, flows=None, risk: RiskTable | None = None) -> FastAPI:
    """No keys configured = open server (local default). `media` enables images/audio in the state (perception-service).
    `flows` stores saved flows (an object with `async get(name) -> dict | None`, e.g. flows.RedisFlowStore)."""
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
    swarm = worker.engine if hasattr(worker.engine, "request_reload") else None
    batch_max = int(os.environ.get("DRAGONFLY_BATCH_MAX", "10000"))
    batch_concurrency = int(os.environ.get("DRAGONFLY_BATCH_CONCURRENCY", "512"))
    batch_jobs_kept = 20

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

    async def decide(req: DecideRequest, request: Request, background: BackgroundTasks, stream: bool = False):
        no_cache = "no-cache" in request.headers.get("cache-control", "")
        if stream:
            return StreamingResponse(stream_decision(req, request.state.key_digest, background),
                                     media_type="text/event-stream", headers={"cache-control": "no-cache"})
        return await decide_one(req, no_cache, request.state.key_digest, background)

    def cascade_of(engine):
        """The general S -> M cascade behind the worker (directly or inside the swarm), or None."""
        engine = getattr(engine, "general", engine)
        return engine if isinstance(engine, Cascade) else None

    async def stream_decision(req: DecideRequest, key_digest: str | None, background: BackgroundTasks):
        """Anytime answers (SSE): tier S's answers at once, tier M's upgrades for the unsure ones, then the final body
        (after plugins). A caller in a real-time loop acts on the first event; later events only refine it."""
        def event(name: str, data: dict) -> str:
            payload = json.dumps(data, separators=(",", ":"))
            return f"event: {name}\ndata: {payload}\n\n"

        cascade = cascade_of(worker.engine)
        if cascade is None or find_media(req.state) or (swarm is not None and req.model not in MODEL_NAMES):
            try:  # nothing to refine (no cascade, a specialist, or media to resolve first): one final event
                yield event("done", await decide_one(req, True, key_digest, background))
            except HTTPException as e:
                yield event("error", {"status": e.status_code, "detail": e.detail})
            return
        started = time.perf_counter()
        try:
            req = await plugins.aon_request(req)
            req, two_stage, stage1_tokens = await prune_large_choices(req, False)
            record, meta = to_record(req)
            probs, stats = await asyncio.wrap_future(worker.submit({**record, "no_escalate": True}))
            answers = to_answers(probs, meta)
            for m in meta:
                answers[m["id"]]["tier"] = "S"
            yield event("answers", {"answers": answers, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                                    "final": False})
            hard = [j for j, (q, p) in enumerate(zip(record["questions"], probs))
                    if question_confidence(q["qtype"], p) < cascade.threshold_for(q["qtype"])]
            tokens = stats["tokens"] + stage1_tokens
            if hard:
                sub = {"state": record["state"], "questions": [record["questions"][j] for j in hard], "force_large": True}
                probs2, stats2 = await asyncio.wrap_future(worker.submit(sub))
                tokens += stats2["tokens"]
                for j, p in zip(hard, probs2):
                    probs[j] = p
                upgraded = to_answers([probs[j] for j in hard], [meta[j] for j in hard])
                for a in upgraded.values():
                    a["tier"] = "M"
                answers.update(upgraded)
                yield event("update", {"answers": upgraded, "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                                       "final": False})
            if req.max_error is not None:
                apply_risk(req, answers, probs, meta, None)
            for qid, info in two_stage.items():
                answers[qid]["two_stage"] = info
            body = {"model": req.model, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0},
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2), "cached": False, "final": True}
            body = await plugins.aon_decision(req, body)
            background.add_task(keys.record_usage, key_digest, len(meta), tokens)
            REQUESTS.labels("ok").inc()
            yield event("done", body)
        except (PluginError, MediaError) as e:
            yield event("error", {"status": e.status, "detail": str(e)})
        except HTTPException as e:
            yield event("error", {"status": e.status_code, "detail": e.detail})
        except ValueError as e:
            yield event("error", {"status": 422, "detail": str(e)})

    async def decide_one(req: DecideRequest, no_cache: bool, key_digest: str | None,
                         background: BackgroundTasks, low_priority: bool = False) -> dict:
        """The whole decision pipeline (media, plugins, cache, model), shared by /v1/systemone and flow steps."""
        started = time.perf_counter()
        media_ms = 0.0
        try:
            if find_media(req.state):
                if media is None:
                    raise MediaError("this server has no perception-service (PERCEPTION_URL): images and audio are not supported")
                state, media_ms = await media.resolve(req.state)
                req = req.model_copy(update={"state": state})
            req = await plugins.aon_request(req)
            req, two_stage, stage1_tokens = await prune_large_choices(req, low_priority)
            record, meta = to_record(req)
            if swarm is not None and req.model not in MODEL_NAMES:
                record["specialist"] = req.model
            version = swarm.version(req.model) if swarm is not None else ""
            key = record_key(record, req.model + version)  # a retrained specialist never serves old answers
            # "Cache-Control: no-cache" skips the lookup (standard HTTP semantics); the answer is still stored
            hit = None if no_cache else await cache.aget(key)
            CACHE.labels("hit" if hit else "miss").inc()
            if hit:
                probs, stats = hit
                latency = 0.0
            else:
                probs, stats = await asyncio.wrap_future(worker.submit(record, low_priority))
                if swarm is not None:
                    stats["specialist"] = record.get("routed_to", "general")
                await cache.aput(key, (probs, stats))
                latency = stats["latency_ms"]
                MODEL_LATENCY.observe(latency / 1000)
            answers = to_answers(probs, meta)
            for m, tier in zip(meta, stats["tiers"]):
                answers[m["id"]]["tier"] = tier
            if req.max_error is not None:
                apply_risk(req, answers, probs, meta, stats.get("specialist"))
            for qid, info in two_stage.items():
                answers[qid]["two_stage"] = info
            body = {
                "model": req.model,
                "answers": answers,
                "usage": {"input_tokens": stats["tokens"] + stage1_tokens, "output_tokens": 0, "media_ms": media_ms},
                "latency_ms": latency,
                "cached": bool(hit),
            }
            if "specialist" in stats:
                body["specialist"] = stats["specialist"]
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
        background.add_task(keys.record_usage, key_digest, len(meta), stats["tokens"])
        return body

    def apply_risk(req: DecideRequest, answers: dict, probs, meta, specialist: str | None) -> None:
        if risk is None:
            raise HTTPException(400, "max_error needs a fitted risk table on this server (DRAGONFLY_RISK, "
                                     "scripts/fit_risk.py)")
        for m, p in zip(meta, probs):
            answer = answers[m["id"]]
            if specialist not in (None, "general"):
                # the table was fitted on the general model's answers: promise nothing for a specialist
                answer.update(decided=False, set=list(m["keys"]),
                              risk={"max_error": req.max_error, "note": f"no risk table for specialist {specialist}"})
            else:
                risk.apply(answer, p, m["keys"], m["type"], req.max_error)

    async def prune_large_choices(req: DecideRequest, low_priority: bool):
        """Stage 1 of every choice question above MAX_OPTIONS: keep its best FINALISTS options (twostage.py)."""
        big = {qid: q for qid, q in req.questions.items() if q.type == "choice" and len(q.criteria) > MAX_OPTIONS}
        if not big:
            return req, {}, 0
        size = MAX_OPTIONS - 1
        while True:  # a chunk must fit the model's token budget: halve it until it does
            questions, owners = {}, []
            for qid, q in big.items():
                for c, keys in enumerate(twostage.chunks(list(q.criteria), size)):
                    questions[f"{qid}#{c}"] = {"type": "choice", "instructions": q.instructions,
                                               "criteria": {k: q.criteria[k] for k in keys}}
                    owners.append((qid, keys))
            stage1 = DecideRequest.model_validate({"state": req.state, "model": req.model, "questions": questions})
            record, _ = to_record(stage1)
            record["no_escalate"] = True
            if swarm is not None and req.model not in MODEL_NAMES:
                record["specialist"] = req.model
            try:
                probs, stats = await asyncio.wrap_future(worker.submit(record, low_priority))
                break
            except ValueError:
                if size <= 8:
                    raise
                size //= 2
        updated, info = dict(req.questions), {}
        for qid, q in big.items():
            mine = [(keys, p) for (owner, keys), p in zip(owners, probs) if owner == qid]
            top, pruned = twostage.finalists(twostage.global_scores([k for k, _ in mine], [p for _, p in mine]))
            updated[qid] = q.model_copy(update={"criteria": {k: q.criteria[k] for k in top}})
            info[qid] = {"options": len(q.criteria), "candidates": len(top), "pruned_mass": round(pruned, 4)}
        return req.model_copy(update={"questions": updated}), info, stats["tokens"]

    app.post("/v1/systemone")(decide)
    app.post("/v1/decide")(decide)

    # ---- batch / map-reduce ----------------------------------------------------------------------------------------
    jobs: dict[str, dict] = {}

    async def run_batch(requests: list, no_cache: bool, key_digest: str | None, background: BackgroundTasks,
                        progress: dict | None = None) -> list[dict]:
        """Every request through the normal pipeline at batch priority, shortest first (less padding per forward
        pass), with bounded concurrency. Results come back in the original order; a bad request yields
        {"error": ...} instead of failing the batch."""
        results: list[dict | None] = [None] * len(requests)
        parsed: list[tuple[int, DecideRequest]] = []
        for i, body in enumerate(requests):
            try:
                parsed.append((i, DecideRequest.model_validate(body)))
            except ValidationError as e:
                results[i] = {"error": f"invalid request: {e.errors()[:2]}"}
        parsed.sort(key=lambda x: len(render(x[1].state)))
        slots = asyncio.Semaphore(batch_concurrency)

        async def one(i: int, req: DecideRequest):
            async with slots:
                try:
                    results[i] = await decide_one(req, no_cache, key_digest, background, low_priority=True)
                except HTTPException as e:
                    results[i] = {"error": e.detail}
                if progress is not None:
                    progress["done"] += 1
        await asyncio.gather(*(one(i, req) for i, req in parsed))
        if progress is not None:
            progress["done"] = len(requests)
        return results

    def batch_requests(payload: dict) -> list:
        requests = payload.get("requests")
        if not isinstance(requests, list) or not requests:
            raise HTTPException(422, 'send {"requests": [<a /v1/systemone request>, ...]}')
        if len(requests) > batch_max:
            raise HTTPException(413, f"at most {batch_max} requests per batch (DRAGONFLY_BATCH_MAX)")
        return requests

    @app.post("/v1/batch")
    async def batch(payload: dict, request: Request, background: BackgroundTasks):
        """Many decisions in one call ("map-reduce over data"): {"requests": [...]} -> {"results": [...], "stats"}.
        Runs at batch priority: live requests always go first."""
        requests = batch_requests(payload)
        started = time.perf_counter()
        results = await run_batch(requests, "no-cache" in request.headers.get("cache-control", ""),
                                  request.state.key_digest, background)
        return {"results": results, "stats": batch_stats(results, time.perf_counter() - started)}

    @app.post("/v1/batch/jobs")
    async def start_batch_job(payload: dict, request: Request, background: BackgroundTasks):
        """The same as /v1/batch, in the background: returns {"id"} at once; poll GET /v1/batch/jobs/{id}."""
        requests = batch_requests(payload)
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "status": "running", "total": len(requests), "done": 0, "created": time.time()}
        jobs[job_id] = job
        for old in sorted(jobs.values(), key=lambda j: j["created"])[:-batch_jobs_kept]:
            jobs.pop(old["id"], None)
        key_digest = request.state.key_digest
        no_cache = "no-cache" in request.headers.get("cache-control", "")

        async def work():
            started = time.perf_counter()
            try:
                job["results"] = await run_batch(requests, no_cache, key_digest, background_usage, job)
                job["stats"] = batch_stats(job["results"], time.perf_counter() - started)
                job["status"] = "done"
            except Exception as e:  # noqa: BLE001 - a job must end in a state the client can read
                job.update(status="failed", error=str(e)[:300])
        job["task"] = asyncio.create_task(work())
        return {"id": job_id, "total": len(requests)}

    @app.get("/v1/batch/jobs/{job_id}")
    def batch_job(job_id: str, results: bool = True):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such batch job on this replica (jobs live in the replica that started them)")
        out = {k: v for k, v in job.items() if k != "task" and (results or k != "results")}
        return out

    class _Usage:
        """Background jobs record usage right away (there is no response to attach a BackgroundTasks to)."""

        def add_task(self, fn, *args):
            asyncio.get_running_loop().create_task(fn(*args))
    background_usage = _Usage()

    async def run(flow: dict, payload: dict, request: Request, background: BackgroundTasks) -> dict:
        no_cache = "no-cache" in request.headers.get("cache-control", "")

        async def step(body: dict) -> dict:
            try:
                req = DecideRequest.model_validate(body)
            except ValidationError as e:
                raise HTTPException(422, f"flow step request is invalid: {e.errors()[:3]}") from e
            return await decide_one(req, no_cache, request.state.key_digest, background)
        try:
            out = await run_flow(flow, payload.get("state"), step)
        except FlowError as e:
            raise HTTPException(422, str(e)) from e
        if payload.get("mermaid"):
            out["mermaid"] = mermaid(flow, out["path"])
        return out

    @app.post("/v1/flows/run")
    async def run_inline_flow(payload: dict, request: Request, background: BackgroundTasks):
        """Run a flow given inline: {"flow": {...}, "state": ..., "mermaid": true?}. See dragonfly/flows.py."""
        if not isinstance(payload.get("flow"), dict):
            raise HTTPException(422, "send {\"flow\": {...}, \"state\": ...}")
        return await run(payload["flow"], payload, request, background)

    @app.post("/v1/flows/{name}/run")
    async def run_saved_flow(name: str, payload: dict, request: Request, background: BackgroundTasks):
        """Run a saved flow (created in the UI, stored by the backend): {"state": ..., "mermaid": true?}."""
        flow = await flows.get(name) if flows is not None else None
        if flow is None:
            raise HTTPException(404, f"no flow named {name!r}")
        return await run(flow, payload, request, background)

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
        out = [{"name": n, **card} for n in MODEL_NAMES]
        if swarm is not None:
            out += [{"name": "auto", "description": "routes each request to the best specialist"}]
            out += [{"name": s["name"], "description": s["description"], "tier": s["tier"], "specialist": True}
                    for s in swarm.list()]
        return {"models": out}

    @app.get("/v1/specialists")
    def specialists():
        """The swarm: specialists this server can route to (the `model` field of a request)."""
        return {"specialists": swarm.list() if swarm is not None else []}

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "trained": worker.engine.describe()["trained"]}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
