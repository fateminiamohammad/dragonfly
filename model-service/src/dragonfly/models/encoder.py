"""Dragonfly-S: the speed tier. A bidirectional encoder (ModernBERT by default) reads one question per row:

    [CLS] instructions  - option 1  - option 2 ... [SEP] state [SEP]

and a pointer head scores every option against the question in the same forward pass. The query is the [CLS]
vector; each option's key is the mean of its tokens. Options come before the state, so truncation only ever cuts the
state and never an option.
"""

from __future__ import annotations

import math

import torch
from torch import nn

OPTION_PREFIX = "\n- "


class PointerHead(nn.Module):
    """Scaled dot product between a question query and one key per option, as in Kev."""

    def __init__(self, hidden: int, dim: int = 256):
        super().__init__()
        self.q = nn.Linear(hidden, dim)
        self.k = nn.Linear(hidden, dim)
        self.scale = 1 / math.sqrt(dim)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        # query (R, H), keys (R, K, H) -> logits (R, K)
        return torch.einsum("rd,rkd->rk", self.q(query), self.k(keys)) * self.scale


class EncoderDecider(nn.Module):
    def __init__(self, backbone: nn.Module, head_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.head = PointerHead(backbone.config.hidden_size, head_dim)

    def encode(self, tokenizer, records: list[dict], max_length: int):
        """records -> (model inputs with one row per question, input tokens per record)."""
        rows, states, owners = [], [], []
        for i, rec in enumerate(records):
            for q in rec["questions"]:
                rows.append(q)
                states.append(rec["state"])
                owners.append(i)
        batch = encode_rows(tokenizer, rows, states, max_length)
        tokens = [0] * len(records)
        for owner, n in zip(owners, batch["attention_mask"].sum(1).tolist()):
            tokens[owner] += n
        return batch, tokens

    def forward(self, input_ids, attention_mask, option_mask) -> torch.Tensor:
        """option_mask (R, K, L) marks each option's tokens. Returns logits (R, K); absent options get -inf."""
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        om = option_mask.to(hidden.dtype)
        counts = om.sum(-1)
        keys = torch.einsum("rkl,rlh->rkh", om, hidden) / counts.clamp(min=1).unsqueeze(-1)
        logits = self.head(hidden[:, 0], keys).float()
        return logits.masked_fill(counts == 0, float("-inf"))


def question_text(instr: str, options: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """-> the question block and each option's character span within it."""
    text, spans = instr, []
    for opt in options:
        text += OPTION_PREFIX
        spans.append((len(text), len(text) + len(opt)))
        text += opt
    return text, spans


def encode_rows(tokenizer, rows: list[dict], state_texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    """Tokenize question rows against their states. Each row is {"instr", "options"}.

    Raises ValueError when a question block does not fit or an option ends up with no tokens.
    """
    blocks = [question_text(r["instr"], r["options"]) for r in rows]
    try:
        enc = tokenizer(
            [b[0] for b in blocks],
            state_texts,
            truncation="only_second",
            max_length=max_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
    except Exception as e:  # the tokenizer refuses when the question block alone exceeds max_length
        raise ValueError(f"question and options do not fit in {max_length} tokens: {e}") from e

    offsets = enc.pop("offset_mapping")
    n_rows, length = enc["input_ids"].shape
    n_opts = max(len(b[1]) for b in blocks)
    option_mask = torch.zeros(n_rows, n_opts, length, dtype=torch.bool)
    for r, (_, spans) in enumerate(blocks):
        in_question = torch.tensor([s == 0 for s in enc.sequence_ids(r)])
        tok_start, tok_end = offsets[r, :, 0], offsets[r, :, 1]
        real = in_question & (tok_end > tok_start)
        for k, (s, e) in enumerate(spans):
            m = real & (tok_start < e) & (tok_end > s)
            if not m.any():
                raise ValueError(f"option {k} of a question has no tokens (empty or truncated)")
            option_mask[r, k] = m
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "option_mask": option_mask}
