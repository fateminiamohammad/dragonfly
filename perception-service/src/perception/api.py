"""perception-service HTTP API. Internal service: the model-service calls it (and proxies POST /v1/perceive to clients
with API-key auth); it is not exposed by nginx.

Configuration by environment:
  PERCEPTION_DEVICE          cuda | cpu (default: cuda when available)
  PERCEPTION_ASR_MODEL_EN    default nvidia/parakeet-ctc-0.6b (1.1b: same text on our sample, 1.7x slower)
  PERCEPTION_ASR_MODEL_FA    default jonatasgrosman/wav2vec2-large-xlsr-53-persian
  PERCEPTION_ASR_MODELS      extra languages: "de=org/model,ar=org/model"
  PERCEPTION_TAGS_MODEL      default google/siglip2-base-patch16-224
  PERCEPTION_TAGS_FILE       tag vocabulary, one per line (default: built-in)
  PERCEPTION_OCR_LANGS       OCR recognition models to allow, e.g. "en,arabic" (default "en")
  PERCEPTION_PRELOAD         models to load at startup, e.g. "tags,ocr:en,asr:en" (default); others load on first use
  PERCEPTION_MAX_MB          per item (default 20)
  PERCEPTION_MAX_AUDIO_S     per audio item (default 60)
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import threading
import time
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from . import __version__
from .audio import CTCRecognizer, decode_audio
from .ocr import OCR
from .tags import Tagger, decode_image, load_tags

log = logging.getLogger("perception")

ITEMS = Counter("perception_items_total", "Media items processed", ["type", "status"])
STAGE = Histogram("perception_stage_seconds", "Time per processing stage per request", ["stage"],
                  buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))

DEFAULT_ASR = {"en": "nvidia/parakeet-ctc-0.6b", "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian"}


class Item(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    type: Literal["image", "audio"]
    data: str = Field(description="base64-encoded file")
    tasks: list[Literal["ocr", "tags"]] = ["ocr", "tags"]  # images
    language: str = "en"  # audio: which speech model
    ocr_lang: str = "en"  # images: which OCR recognition model


class PerceiveRequest(BaseModel):
    items: list[Item] = Field(min_length=1, max_length=16)


class Perception:
    """Owns the models. Heavy models load lazily (or at startup via preload); one lock serializes GPU work."""

    def __init__(self, device: str, asr_models: dict[str, str], tags_model: str, tags: list[str],
                 ocr_langs: list[str], max_mb: float = 20, max_audio_s: float = 60,
                 tagger: Tagger | None = None, ocr: dict | None = None, recognizers: dict | None = None):
        self.device = device
        self.asr_models = asr_models
        self.tags_model = tags_model
        self.tag_list = tags
        self.ocr_langs = ocr_langs
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.max_audio_s = max_audio_s
        self._tagger = tagger
        self._ocr = dict(ocr or {})
        self._asr = dict(recognizers or {})
        self.lock = threading.Lock()

    @classmethod
    def from_env(cls) -> Perception:
        asr = dict(DEFAULT_ASR)
        asr["en"] = os.environ.get("PERCEPTION_ASR_MODEL_EN", asr["en"])
        asr["fa"] = os.environ.get("PERCEPTION_ASR_MODEL_FA", asr["fa"])
        for pair in filter(None, os.environ.get("PERCEPTION_ASR_MODELS", "").split(",")):
            lang, model = pair.split("=", 1)
            asr[lang.strip()] = model.strip()
        device = os.environ.get("PERCEPTION_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(device, asr, os.environ.get("PERCEPTION_TAGS_MODEL", "google/siglip2-base-patch16-224"),
                   load_tags(os.environ.get("PERCEPTION_TAGS_FILE")),
                   [x.strip() for x in os.environ.get("PERCEPTION_OCR_LANGS", "en").split(",") if x.strip()],
                   float(os.environ.get("PERCEPTION_MAX_MB", "20")), float(os.environ.get("PERCEPTION_MAX_AUDIO_S", "60")))

    # ---- lazy models ----------------------------------------------------------------------------------------------
    def tagger(self) -> Tagger:
        if self._tagger is None:
            self._tagger = Tagger(self.tags_model, self.device, self.tag_list)
        return self._tagger

    def ocr(self, lang: str) -> OCR:
        if lang not in self.ocr_langs:
            raise ValueError(f"OCR language {lang!r} is not enabled (PERCEPTION_OCR_LANGS={','.join(self.ocr_langs)})")
        if lang not in self._ocr:
            self._ocr[lang] = OCR(lang, use_cuda=self.device == "cuda")
        return self._ocr[lang]

    def asr(self, lang: str) -> CTCRecognizer:
        if lang not in self.asr_models:
            raise ValueError(f"no speech model for language {lang!r} (available: {', '.join(sorted(self.asr_models))})")
        if lang not in self._asr:
            self._asr[lang] = CTCRecognizer(self.asr_models[lang], self.device)
        return self._asr[lang]

    def preload(self, spec: str) -> None:
        for part in filter(None, (s.strip() for s in spec.split(","))):
            kind, _, lang = part.partition(":")
            if kind == "tags":
                self.tagger()
            elif kind == "ocr":
                self.ocr(lang or "en")
            elif kind == "asr":
                self.asr(lang or "en")
            log.info("preloaded %s", part)
        self.warm()

    def warm(self) -> None:
        """Run each loaded model once on dummy input: first-use GPU setup (kernel selection, CUDA graph capture) cost
        ~3.7 s on the first real request before this existed."""
        import numpy as np
        from PIL import Image

        blank = Image.new("RGB", (640, 480), "white")
        if self._tagger is not None:
            for n in (1, 2, 4):
                self._tagger.tag([blank] * n)
        for ocr in self._ocr.values():
            ocr.read(blank)
        for asr in self._asr.values():
            asr.transcribe([np.zeros(16_000 * 5, dtype=np.float32)])
        log.info("warmed up loaded models")

    def describe(self) -> dict:
        return {"device": self.device, "asr_models": self.asr_models, "tags_model": self.tags_model,
                "tags": len(self.tag_list), "ocr_langs": self.ocr_langs,
                "loaded": {"tags": self._tagger is not None, "ocr": sorted(self._ocr), "asr": sorted(self._asr)}}

    # ---- work -----------------------------------------------------------------------------------------------------
    def decode(self, item: Item) -> bytes:
        if len(item.data) * 3 // 4 > self.max_bytes:
            raise ValueError(f"item {item.id!r} is larger than {self.max_bytes // (1024 * 1024)} MB")
        try:
            return base64.b64decode(item.data, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"item {item.id!r}: data is not valid base64") from e

    def perceive(self, items: list[Item]) -> tuple[dict, dict]:
        """-> (results by item id, milliseconds by stage). Raises ValueError (-> 422) for bad input."""
        if len({i.id for i in items}) != len(items):
            raise ValueError("item ids must be unique")
        timings: dict[str, float] = {}

        def timed(stage: str, fn):
            t = time.perf_counter()
            out = fn()
            ms = (time.perf_counter() - t) * 1000
            timings[stage] = round(timings.get(stage, 0) + ms, 2)
            STAGE.labels(stage).observe(ms / 1000)
            return out

        images, audios = [], []
        for item in items:
            raw = self.decode(item)
            try:
                if item.type == "image":
                    images.append((item, timed("decode", lambda raw=raw: decode_image(raw))))
                else:
                    audios.append((item, timed("decode", lambda raw=raw: decode_audio(raw, self.max_audio_s))))
            except ValueError as e:
                ITEMS.labels(item.type, "invalid").inc()
                raise ValueError(f"item {item.id!r}: {e}") from e

        results: dict[str, dict] = {item.id: {"type": item.type} for item in items}
        with self.lock:
            tag_items = [(it, img) for it, img in images if "tags" in it.tasks]
            if tag_items:
                tags = timed("tags", lambda: self.tagger().tag([img for _, img in tag_items]))
                for (it, _), t in zip(tag_items, tags):
                    results[it.id]["tags"] = t
            for it, img in images:
                if "ocr" in it.tasks:
                    results[it.id]["ocr"] = timed("ocr", lambda it=it, img=img: self.ocr(it.ocr_lang).read(img))
            by_lang: dict[str, list] = {}
            for it, wav in audios:
                by_lang.setdefault(it.language, []).append((it, wav))
            for lang, group in by_lang.items():
                texts = timed("asr", lambda lang=lang, group=group: self.asr(lang).transcribe([w for _, w in group]))
                for (it, wav), text in zip(group, texts):
                    results[it.id].update({"transcript": text, "language": lang,
                                           "duration_s": round(wav.size / 16_000, 2)})
        for item in items:
            ITEMS.labels(item.type, "ok").inc()
        return results, timings


def create_app(perception: Perception) -> FastAPI:
    app = FastAPI(title="dragonfly-perception", version=__version__)

    @app.post("/v1/perceive")
    def perceive(req: PerceiveRequest):
        started = time.perf_counter()
        try:
            results, timings = perception.perceive(req.items)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        timings["total"] = round((time.perf_counter() - started) * 1000, 2)
        return {"items": results, "latency_ms": timings}

    @app.get("/v1/info")
    def info():
        return perception.describe()

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def build_app() -> FastAPI:
    logging.basicConfig(level=os.environ.get("PERCEPTION_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    perception = Perception.from_env()
    perception.preload(os.environ.get("PERCEPTION_PRELOAD", "tags,ocr:en,asr:en"))
    return create_app(perception)


def main() -> None:
    import uvicorn

    uvicorn.run(build_app(), host=os.environ.get("PERCEPTION_HOST", "0.0.0.0"),
                port=int(os.environ.get("PERCEPTION_PORT", "8001")), log_level="info", access_log=False)


if __name__ == "__main__":
    main()
