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
    """Scaled dot product between a question query and one key per option, as in Kev.

    norm=True puts a LayerNorm on the hidden states before the projections, so the logit scale no longer depends on
    the backbone's activation magnitudes. Without it, Qwen3-4B's larger hidden states blew the logits up mid-training
    (batch loss 17 at step 1,000, above log(255) = 5.5, the loss of a uniform guess)."""

    def __init__(self, hidden: int, dim: int = 256, norm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(hidden) if norm else nn.Identity()
        self.q = nn.Linear(hidden, dim)
        self.k = nn.Linear(hidden, dim)
        self.scale = 1 / math.sqrt(dim)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        # query (R, H), keys (R, K, H) -> logits (R, K)
        return torch.einsum("rd,rkd->rk", self.q(self.norm(query)), self.k(self.norm(keys))) * self.scale


class EncoderDecider(nn.Module):
    def __init__(self, backbone: nn.Module, head_dim: int = 256, head_norm: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = PointerHead(backbone.config.hidden_size, head_dim, head_norm)

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


# ---- packed tier S: one sequence per request (the state is read once) ------------------------------------------------
PACKED_OPTION = " - {}"


def packed_masks(qid: torch.Tensor, opt: torch.Tensor, pos: torch.Tensor, window: int | None) -> dict[str, torch.Tensor]:
    """Boolean (B, 1, L, L) masks, True = may attend, for ModernBERT's two layer types.

    Bidirectional, but by segment: the state sees only the state (so it is read once, the same for every question);
    a question's text sees the state and itself; an option sees the state, its question and itself, never another
    option. Padding (qid -2) sees only itself. Local layers additionally keep |position_i - position_j| <= window."""
    qi, qj = qid.unsqueeze(2), qid.unsqueeze(1)
    oi, oj = opt.unsqueeze(2), opt.unsqueeze(1)
    allow = (qj == -1) | ((qj == qi) & ((oj == -1) | (oj == oi)))
    allow = allow & (qi != -2) & (qj != -2)
    allow = allow | torch.eye(qid.shape[1], dtype=torch.bool, device=qid.device)
    local = allow
    if window is not None:
        local = allow & ((pos.unsqueeze(2) - pos.unsqueeze(1)).abs() <= window)
        local = local | torch.eye(qid.shape[1], dtype=torch.bool, device=qid.device)
    return {"full_attention": allow.unsqueeze(1), "sliding_attention": local.unsqueeze(1)}


def pack_encoder(tokenizer, record: dict, max_length: int) -> dict:
    """[CLS] state [SEP] | [CLS] q1 [SEP] - opt1 - opt2 | [CLS] q2 [SEP] ... as token ids with segment ids.

    Every question block starts at the same position (after the state), and every option of a question at the same
    position, so neither the order of the questions nor of the options changes a score. The state is truncated so the
    whole record fits in max_length; ValueError when the questions alone do not fit."""
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    blocks = []
    for qi, q in enumerate(record["questions"]):
        head = [cls] + tokenizer(q["instr"], add_special_tokens=False).input_ids + [sep]
        opts = [tokenizer(PACKED_OPTION.format(o), add_special_tokens=False).input_ids for o in q["options"]]
        if any(not o for o in opts):
            raise ValueError(f"an option of question {qi} has no tokens")
        blocks.append((head, opts))
    needed = sum(len(h) + sum(len(o) for o in opts) for h, opts in blocks)
    room = max_length - needed - 2
    if room < 1:
        raise ValueError(f"the questions and options alone need {needed} tokens (max {max_length})")
    state = tokenizer(record["state"], add_special_tokens=False).input_ids[:room]
    ids = [cls] + state + [sep]
    n = len(ids)
    pos, qid, opt = list(range(n)), [-1] * n, [-1] * n
    query, keys = [], []
    for qi, (head, opts) in enumerate(blocks):
        query.append(len(ids))
        ids += head
        pos += list(range(n, n + len(head)))
        qid += [qi] * len(head)
        opt += [-1] * len(head)
        start = n + len(head)
        spans = []
        for oi, o in enumerate(opts):
            spans.append(list(range(len(ids), len(ids) + len(o))))
            ids += o
            pos += list(range(start, start + len(o)))
            qid += [qi] * len(o)
            opt += [oi] * len(o)
        keys.append(spans)
    return {"ids": ids, "pos": pos, "qid": qid, "opt": opt, "query": query, "keys": keys}


class PackedEncoderDecider(EncoderDecider):
    """Tier S with one packed sequence per request: a document with Q questions costs one pass over the document
    instead of Q. Query = the question's [CLS]; key = the mean of the option's tokens. Same forward/pointer interface
    as tier M's DecoderDecider, so the same CUDA-graph runner serves both."""

    def masks(self, qid, opt, position_ids):
        window = getattr(self.backbone.config, "local_attention", None)
        return packed_masks(qid, opt, position_ids, window // 2 if window else None)

    def encode(self, tokenizer, records: list[dict], max_length: int):
        packed = [pack_encoder(tokenizer, r, max_length) for r in records]
        length = max(len(p["ids"]) for p in packed)
        b = len(packed)
        ids = torch.full((b, length), tokenizer.pad_token_id or 0, dtype=torch.long)
        pos = torch.zeros((b, length), dtype=torch.long)
        qid = torch.full((b, length), -2, dtype=torch.long)
        opt = torch.full((b, length), -1, dtype=torch.long)
        queries, keys = [], []
        for r, p in enumerate(packed):
            n = len(p["ids"])
            ids[r, :n], pos[r, :n] = torch.tensor(p["ids"]), torch.tensor(p["pos"])
            qid[r, :n], opt[r, :n] = torch.tensor(p["qid"]), torch.tensor(p["opt"])
            queries += [r * length + i for i in p["query"]]
            keys += [[[r * length + i for i in span] for span in q] for q in p["keys"]]
        n_opts = max(len(k) for k in keys)
        n_tok = max(len(span) for k in keys for span in k)
        key_idx = torch.full((len(keys), n_opts, n_tok), -1, dtype=torch.long)
        for i, k in enumerate(keys):
            for j, span in enumerate(k):
                key_idx[i, j, : len(span)] = torch.tensor(span)
        batch = {"input_ids": ids, "position_ids": pos, "qid": qid, "opt": opt,
                 "query_idx": torch.tensor(queries), "key_idx": key_idx}
        return batch, [len(p["ids"]) for p in packed]

    def forward(self, input_ids, position_ids, qid, opt, query_idx, key_idx) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, position_ids=position_ids,
                               attention_mask=self.masks(qid, opt, position_ids)).last_hidden_state
        return self.pointer(hidden.reshape(-1, hidden.shape[-1]), query_idx, key_idx)

    def pointer(self, flat_hidden: torch.Tensor, query_idx: torch.Tensor, key_idx: torch.Tensor) -> torch.Tensor:
        """key_idx (Q, K, T): each option's token indices (-1 = none); its key is their mean."""
        dtype = self.head.q.weight.dtype
        valid = (key_idx >= 0).to(dtype)
        tokens = flat_hidden[key_idx.clamp(min=0)].to(dtype) * valid.unsqueeze(-1)
        counts = valid.sum(-1)
        keys = tokens.sum(2) / counts.clamp(min=1).unsqueeze(-1)
        logits = self.head(flat_hidden[query_idx].to(dtype), keys).float()
        return logits.masked_fill(counts == 0, float("-inf"))
