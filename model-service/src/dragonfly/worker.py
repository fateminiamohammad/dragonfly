"""Trainer worker: `dragonfly-worker`. Trains specialists that users upload (docs/SWARM.md, UI Specialists page).

The backend validates an upload, stores it as labelled JSONL under DRAGONFLY_UPLOADS and pushes a job:

    LPUSH dragonfly:train-jobs '{"id": "...", "op": "train", "name": "invoices", "description": "...", "tier": "S",
                                 "data": "/runs/uploads/<id>.jsonl", "epochs": 3}'
    LPUSH dragonfly:train-jobs '{"id": "...", "op": "delete", "name": "invoices"}'

For each job the worker splits the data (70% train, 10% calibration, 20% held-out test, deterministic), warm-starts
from the general model of that tier (a specialist learns its job on top of everything the general model knows),
trains, fits the temperature, evaluates on the held-out 20%, writes the specialist into DRAGONFLY_SPECIALISTS and
publishes on dragonfly:specialists so every model-service replica reloads. Progress is kept in the hash
dragonfly:train-job:<id> (status, step, total, loss, metrics, error), which the backend shows in the UI.

GPU sharing: a tier S job fits beside serving (a few GB). A tier M job trains adapters on the 4B backbone and needs
most of a 24 GB GPU, so it is refused unless DRAGONFLY_TRAIN_M=1 (a second GPU, or tier M serving stopped).

Environment:
  REDIS_URL                 required
  DRAGONFLY_SPECIALISTS     where specialists are written (default /runs/specialists)
  DRAGONFLY_UPLOADS         where the backend stores uploads (default /runs/uploads)
  DRAGONFLY_INIT_S          warm start for tier S jobs (default: DRAGONFLY_CHECKPOINT)
  DRAGONFLY_INIT_M          warm start for tier M jobs (default: DRAGONFLY_CHECKPOINT_M)
  DRAGONFLY_TRAIN_M         1 = accept tier M jobs
  DRAGONFLY_TRAIN_MAX_LENGTH  tokens per question row while training (default 512: keeps S jobs small)
  DRAGONFLY_TRAIN_BATCH     batch size (default 16)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import time
import traceback
from pathlib import Path

log = logging.getLogger("dragonfly.worker")

JOBS = "dragonfly:train-jobs"
JOB = "dragonfly:train-job:{}"
RELOAD_CHANNEL = "dragonfly:specialists"
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{1,39}$")
RESERVED = {"auto", "general", "dragonfly-latest", "jev-latest", "kev-latest"}
MIN_RECORDS = 50


def split(lines: list[str], seed: int = 0) -> tuple[list[str], list[str], list[str]]:
    """70 / 10 / 20 (train / calibration / test), deterministic."""
    lines = list(lines)
    random.Random(seed).shuffle(lines)
    n_test = max(1, round(len(lines) * 0.2))
    n_cal = max(1, round(len(lines) * 0.1))
    return lines[n_test + n_cal:], lines[n_test:n_test + n_cal], lines[:n_test]


class TrainerWorker:
    def __init__(self, redis, specialists: str, uploads: str, init: dict[str, str | None], allow_m: bool = False,
                 max_length: int = 512, batch_size: int = 16, device: str | None = None, train_fn=None):
        self.redis = redis
        self.specialists = Path(specialists)
        self.uploads = Path(uploads)
        self.init = init
        self.allow_m = allow_m
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        if train_fn is None:
            from .training.train import train as train_fn
        self.train_fn = train_fn

    def update(self, job_id: str, **fields) -> None:
        self.redis.hset(JOB.format(job_id), mapping={k: json.dumps(v) for k, v in fields.items()})

    def run_once(self, timeout: int = 0) -> bool:
        """Take one job (blocking up to timeout seconds; 0 = forever). Returns False when there was none."""
        item = self.redis.brpop(JOBS, timeout=timeout)
        if item is None:
            return False
        job = json.loads(item[1])
        job_id = job.get("id", "unknown")
        try:
            if job.get("op") == "delete":
                self.delete(job["name"])
                self.update(job_id, status="done")
            else:
                self.train(job)
        except Exception as e:
            log.error("job %s failed: %s", job_id, traceback.format_exc())
            self.update(job_id, status="failed", error=str(e)[:500], finished=time.time())
        return True

    def delete(self, name: str) -> None:
        if not NAME.match(name):
            raise ValueError(f"invalid specialist name {name!r}")
        target = self.specialists / name
        if target.exists():
            shutil.rmtree(target)
        self.redis.publish(RELOAD_CHANNEL, json.dumps({"deleted": name}))

    def train(self, job: dict) -> None:
        job_id, name, tier = job["id"], job["name"], job.get("tier", "S")
        if not NAME.match(name) or name in RESERVED:
            raise ValueError(f"invalid specialist name {name!r}: 2-40 lowercase letters, digits and dashes")
        if tier not in ("S", "M"):
            raise ValueError("tier must be S or M")
        if tier == "M" and not self.allow_m:
            raise ValueError("tier M training is disabled on this server (DRAGONFLY_TRAIN_M=1 needs a free GPU)")
        init = self.init.get(tier)
        if not init or not Path(init).is_dir():
            raise ValueError(f"no trained tier {tier} checkpoint to start from (DRAGONFLY_INIT_{tier})")
        data = Path(job["data"])
        if self.uploads.resolve() not in data.resolve().parents:
            raise ValueError("the data file must be inside the uploads folder")
        lines = [ln for ln in data.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) < MIN_RECORDS:
            raise ValueError(f"need at least {MIN_RECORDS} labelled examples, got {len(lines)}")

        self.update(job_id, status="running", step=0, total=0, started=time.time())
        work = self.uploads / f"{job_id}.work"
        work.mkdir(parents=True, exist_ok=True)
        parts = dict(zip(("train", "calibration", "test"), split(lines)))
        for part, rows in parts.items():
            (work / f"{part}.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
        staging = self.specialists / f".{name}.{job_id}"
        args = argparse.Namespace(
            train=str(work / "train.jsonl"), calibration=str(work / "calibration.jsonl"), test=str(work / "test.jsonl"),
            out=str(staging), tier=tier, init=init, backbone=None, head_dim=256, max_length=self.max_length, lora_r=16,
            grad_checkpointing=tier == "M", epochs=int(job.get("epochs", 3)), batch_size=self.batch_size, lr=None,
            no_head_norm=False, head_lr=None, seed=0, device=self.device, log_every=10)
        report = self.train_fn(args, progress=lambda step, total, loss: self.update(
            job_id, step=step, total=total, loss=round(loss, 4)))

        test = report.get("test", {}).get("overall", {})
        metrics = {"accuracy": test.get("accuracy"), "ece": test.get("ece"), "test_questions": test.get("n"),
                   "train_questions": report.get("train_questions"), "seconds": report.get("seconds")}
        card = {"name": name, "description": job.get("description") or name, "metrics": metrics,
                "created": time.strftime("%Y-%m-%d %H:%M"), "job": job_id, "started_from": init}
        (staging / "specialist.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
        target = self.specialists / name
        if target.exists():
            shutil.rmtree(target)  # retraining replaces the old version
        staging.rename(target)
        shutil.rmtree(work, ignore_errors=True)
        self.redis.publish(RELOAD_CHANNEL, json.dumps({"added": name}))
        self.update(job_id, status="done", metrics=metrics, finished=time.time())
        log.info("specialist %s ready: %s", name, metrics)


def main() -> None:
    import redis

    logging.basicConfig(level=os.environ.get("DRAGONFLY_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    worker = TrainerWorker(
        redis.Redis.from_url(os.environ["REDIS_URL"]),
        os.environ.get("DRAGONFLY_SPECIALISTS", "/runs/specialists"),
        os.environ.get("DRAGONFLY_UPLOADS", "/runs/uploads"),
        {"S": os.environ.get("DRAGONFLY_INIT_S") or os.environ.get("DRAGONFLY_CHECKPOINT"),
         "M": os.environ.get("DRAGONFLY_INIT_M") or os.environ.get("DRAGONFLY_CHECKPOINT_M")},
        allow_m=os.environ.get("DRAGONFLY_TRAIN_M") == "1",
        max_length=int(os.environ.get("DRAGONFLY_TRAIN_MAX_LENGTH", "512")),
        batch_size=int(os.environ.get("DRAGONFLY_TRAIN_BATCH", "16")),
        device=os.environ.get("DRAGONFLY_DEVICE") or None,
    )
    Path(worker.specialists).mkdir(parents=True, exist_ok=True)
    log.info("waiting for training jobs on %s", JOBS)
    while True:
        try:
            worker.run_once(timeout=20)  # bounded wait: an idle connection must not hit the socket timeout
        except (redis.ConnectionError, redis.TimeoutError) as e:
            log.warning("redis: %s; retrying", e)
            time.sleep(5)


if __name__ == "__main__":
    main()
