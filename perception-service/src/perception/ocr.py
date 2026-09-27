"""Image -> text with PP-OCR through RapidOCR: non-autoregressive. A segmentation network (DBNet) finds text lines in
one pass; a CTC recognizer reads every line in parallel. No token-by-token generation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)
RTL = re.compile(r"[֐-ࣿיִ-﷿ﹰ-﻿]")  # Hebrew, Arabic, Persian, Urdu blocks


@dataclass
class Line:
    text: str
    confidence: float
    box: list[list[float]]  # 4 corner points [[x, y], ...]

    @property
    def top(self) -> float:
        return min(p[1] for p in self.box)

    @property
    def bottom(self) -> float:
        return max(p[1] for p in self.box)

    @property
    def left(self) -> float:
        return min(p[0] for p in self.box)


def is_rtl(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(bool(RTL.match(c)) for c in letters) > len(letters) / 2


def reading_order(lines: list[Line]) -> list[list[Line]]:
    """Group lines into rows (vertical overlap), top to bottom; order each row left-to-right, or right-to-left when
    most of its text is RTL (Persian/Arabic)."""
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda x: x.top):
        for row in rows:
            ref = row[0]
            overlap = min(ref.bottom, line.bottom) - max(ref.top, line.top)
            if overlap > 0.5 * min(ref.bottom - ref.top, line.bottom - line.top):
                row.append(line)
                break
        else:
            rows.append([line])
    for row in rows:
        rtl = is_rtl(" ".join(x.text for x in row))
        row.sort(key=lambda x: x.left, reverse=rtl)
    return rows


def join_text(rows: list[list[Line]]) -> str:
    return "\n".join(" ".join(x.text for x in row) for row in rows)


class OCR:
    """PP-OCR via RapidOCR. `lang` picks the recognition model (e.g. "en", "ch", "arabic" for Persian/Arabic)."""

    def __init__(self, lang: str = "en", use_cuda: bool = False, engine=None):
        self.lang = lang
        self.engine = engine or self._build(lang, use_cuda)

    @staticmethod
    def _build(lang: str, use_cuda: bool):
        # PyTorch engine (same runtime as the rest of the service, GPU when available). Its models are PP-OCRv4:
        # a multilingual detector and per-script recognizers ("en", "arabic" covers Persian/Arabic/Urdu, "latin", ...).
        from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        # use_cls (180-degree text flip detection) is off: upright documents and photos are the common case, and it
        # would add a third network per line
        params = {"Global.log_level": "warning", "Global.use_cls": False, "EngineConfig.torch.use_cuda": use_cuda}
        for stage in ("Det", "Cls", "Rec"):  # Cls is built even when use_cls is off
            params[f"{stage}.engine_type"] = EngineType.TORCH
            params[f"{stage}.ocr_version"] = OCRVersion.PPOCRV4
            params[f"{stage}.model_type"] = ModelType.MOBILE
        params["Det.lang_type"] = LangDet.CH  # the PP-OCRv4 "ch" detector finds text lines in any script
        # PaddleOCR's standard resize: cap the longer side at 960 px. RapidOCR's default instead raises the shorter side
        # to 736 px, which blew a 900x260 invoice up to ~2548x736 and made detection 2x slower for identical text.
        params["Det.limit_type"] = "max"
        params["Det.limit_side_len"] = 960
        params["Rec.lang_type"] = LangRec(lang)
        return RapidOCR(params=params)

    def read(self, image, min_confidence: float = 0.5) -> dict:
        result = self.engine(np.asarray(image))
        boxes = getattr(result, "boxes", None)
        lines = []
        if boxes is not None:
            for box, text, score in zip(boxes, result.txts, result.scores):
                if score >= min_confidence and text.strip():
                    lines.append(Line(text.strip(), round(float(score), 4), np.asarray(box, dtype=float).tolist()))
        rows = reading_order(lines)
        ordered = [x for row in rows for x in row]
        return {"text": join_text(rows), "lines": [{"text": x.text, "confidence": x.confidence, "box": x.box}
                                                    for x in ordered]}
