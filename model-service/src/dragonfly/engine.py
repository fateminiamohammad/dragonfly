"""Loading, saving and running a Dragonfly checkpoint.

A checkpoint is a directory with config.json {"tier", "backbone", ...}, the tokenizer, head.pt, and:
  tier S (encoder):  backbone/   the full fine-tuned encoder
  tier M (decoder):  lora.pt     only the LoRA adapters; the frozen base model is loaded from `backbone` (a hub id)

`base:<hub id>` (tier S) or `base-m:<hub id>` (tier M) loads a pretrained backbone with an untrained head, so the
service runs before any training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from .models.decoder import DecoderDecider, apply_lora, lora_state_dict
from .models.encoder import EncoderDecider
from .schema import question_confidence

log = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    tier: str = "S"
    backbone: str = "answerdotai/ModernBERT-base"
    head_dim: int = 256
    max_length: int = 1024  # tier S: tokens per question row. tier M: tokens per question branch
    max_state: int = 2048  # tier M: state tokens kept
    lora_r: int = 16
    lora_alpha: float = 32
    temperature: float = 1.0
    trained: bool = False


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(config: CheckpointConfig, backbone: torch.nn.Module) -> torch.nn.Module:
    if config.tier == "S":
        return EncoderDecider(backbone, config.head_dim)
    if config.tier == "M":
        apply_lora(backbone, config.lora_r, config.lora_alpha)
        return DecoderDecider(backbone, config.head_dim, config.max_state, config.max_length)
    raise ValueError(f"unknown tier {config.tier!r}")


def load_backbone(name_or_path, tier: str, device: str) -> torch.nn.Module:
    # the frozen decoder runs in bf16 on GPU; the trainable encoder stays fp32 (autocast handles the speed)
    dtype = torch.bfloat16 if tier == "M" and device == "cuda" else None
    return AutoModel.from_pretrained(name_or_path, dtype=dtype) if dtype else AutoModel.from_pretrained(name_or_path)


class Engine:
    """Runs requests through the model. Call from one thread only (see batching.Worker)."""

    def __init__(self, model: torch.nn.Module, tokenizer, config: CheckpointConfig, device: str | None = None,
                 max_rows: int = 64):
        self.config = config
        self.config.head_dim = model.head.q.out_features  # the model is the source of truth for what gets saved
        self.device = device or default_device()
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_rows = max_rows  # questions per forward pass
        self.autocast = self.device == "cuda" and torch.cuda.is_bf16_supported()

    # ---- persistence -------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str, device: str | None = None) -> Engine:
        device = device or default_device()
        for prefix, tier in (("base:", "S"), ("base-m:", "M")):
            if path.startswith(prefix):
                config = CheckpointConfig(tier=tier, backbone=path[len(prefix):])
                log.warning("serving %s with an UNTRAINED head: answers are meaningless until you train a checkpoint",
                            config.backbone)
                model = build_model(config, load_backbone(config.backbone, tier, device))
                return cls(model, AutoTokenizer.from_pretrained(config.backbone), config, device)
        root = Path(path)
        config = CheckpointConfig(**json.loads((root / "config.json").read_text()))
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
        source = root / "backbone" if config.tier == "S" else config.backbone
        model = build_model(config, load_backbone(source, config.tier, device))
        model.head.load_state_dict(torch.load(root / "head.pt", map_location="cpu", weights_only=True))
        if config.tier == "M":
            missing = model.backbone.load_state_dict(torch.load(root / "lora.pt", map_location="cpu", weights_only=True),
                                                     strict=False)
            if missing.unexpected_keys:
                raise RuntimeError(f"lora.pt does not match {config.backbone}: {missing.unexpected_keys[:3]}")
        return cls(model, tokenizer, config, device)

    def save(self, path: str) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(asdict(self.config), indent=2))
        self.tokenizer.save_pretrained(root / "tokenizer")
        torch.save(self.model.head.state_dict(), root / "head.pt")
        if self.config.tier == "S":
            self.model.backbone.save_pretrained(root / "backbone")
        else:
            torch.save(lora_state_dict(self.model.backbone), root / "lora.pt")

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    # ---- inference ---------------------------------------------------------------------------------------------
    def encode(self, records: list[dict]):
        """records -> (model inputs, input tokens per record). Questions come out in record order."""
        return self.model.encode(self.tokenizer, records, self.config.max_length)

    def logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """-> (questions, max options) raw logits; absent options are -inf."""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            return self.model(**batch)

    def chunks(self, records: list[dict]) -> list[list[int]]:
        """Group record indices so each forward pass holds at most max_rows questions (a bigger record runs alone)."""
        out, current, n = [], [], 0
        for i, rec in enumerate(records):
            k = len(rec["questions"])
            if current and n + k > self.max_rows:
                out.append(current)
                current, n = [], 0
            current.append(i)
            n += k
        if current:
            out.append(current)
        return out

    @torch.inference_mode()
    def probs(self, records: list[dict]) -> tuple[list[list[list[float]]], list[int], list[list[str]]]:
        """records -> (probabilities per record, per question, per option with temperature applied;
        input tokens per record; the tier that answered each question)."""
        out: list[list[list[float]]] = [[] for _ in records]
        tokens = [0] * len(records)
        for idx in self.chunks(records):
            chunk = [records[i] for i in idx]
            batch, n_tokens = self.encode(chunk)
            p = torch.softmax(self.logits(batch) / self.config.temperature, dim=-1).cpu()
            row = 0
            for i, rec, n in zip(idx, chunk, n_tokens):
                tokens[i] = n
                for q in rec["questions"]:
                    out[i].append(p[row, : len(q["options"])].tolist())
                    row += 1
        return out, tokens, [[self.config.tier] * len(r["questions"]) for r in records]

    def describe(self) -> dict:
        c = self.config
        return {"tier": c.tier, "backbone": c.backbone, "trained": c.trained, "temperature": c.temperature,
                "device": self.device}


class Cascade:
    """Tier S answers every question; questions it is unsure about (calibrated confidence below `threshold`) are
    answered again by tier M. Most traffic pays only S's latency; hard questions get M's quality.

    Confidence is only meaningful when S is calibrated (temperature fitted), which dragonfly-train does."""

    def __init__(self, small: Engine, large: Engine, threshold: float = 0.8):
        self.small, self.large, self.threshold = small, large, threshold
        self.config = small.config
        self.device = small.device
        self.escalated = 0
        self.answered = 0

    def probs(self, records: list[dict]):
        probs, tokens, tiers = self.small.probs(records)
        hard = []  # (record index, question index)
        for i, rec in enumerate(records):
            for j, q in enumerate(rec["questions"]):
                if question_confidence(q["qtype"], probs[i][j]) < self.threshold:
                    hard.append((i, j))
        self.answered += sum(len(r["questions"]) for r in records)
        self.escalated += len(hard)
        if hard:
            # one sub-record per original record that has hard questions: M reads each state once
            groups: dict[int, list[int]] = {}
            for i, j in hard:
                groups.setdefault(i, []).append(j)
            sub = [{"state": records[i]["state"], "questions": [records[i]["questions"][j] for j in js]}
                   for i, js in groups.items()]
            p2, t2, _ = self.large.probs(sub)
            for (i, js), ps, n in zip(groups.items(), p2, t2):
                tokens[i] += n
                for j, p in zip(js, ps):
                    probs[i][j] = p
                    tiers[i][j] = self.large.config.tier
        return probs, tokens, tiers

    def describe(self) -> dict:
        return {"tier": "cascade", "threshold": self.threshold, "small": self.small.describe(),
                "large": self.large.describe(), "trained": self.small.config.trained and self.large.config.trained,
                "escalation_rate": round(self.escalated / self.answered, 4) if self.answered else None,
                "device": self.device}
