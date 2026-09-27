"""Dragonfly-M: the quality tier. A decoder LLM (Qwen3 by default), frozen, with small LoRA adapters and a pointer head.

A whole request is packed into ONE sequence and answered in one forward pass:

    [state ...][question 1 ...][option 1a][option 1b]...[question 2 ...][option 2a]...

A custom attention mask decides who sees whom:
  - the state is read once and seen by everything (no re-reading per question);
  - a question sees the state and itself, never another question (answers can't leak between questions);
  - an option sees the state and its own question, never another option.

Every option of a question starts at the SAME position id. The model sees options that are isolated from each other
and all at the same position, so reordering the options cannot change their scores: order invariance by construction,
not by training (Kev flips ~7% of answers under reordering).

The pointer head compares the question's last token (query) with each option's last token (key).
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn

from .encoder import PointerHead

STATE_PREFIX = "State:\n"
QUESTION_TEMPLATE = "\n\nQuestion: {}\nAnswer:"
OPTION_TEMPLATE = " {}\n"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ---- LoRA ----------------------------------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B A x, with W frozen and A, B trained. B starts at zero, so training starts from the base."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        base.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        delta = self.lora_b(self.lora_a(self.dropout(x.to(self.lora_a.weight.dtype))))
        return self.base(x) + (delta * self.scale).to(x.dtype)


def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32, dropout: float = 0.05,
               targets: tuple[str, ...] = LORA_TARGETS) -> int:
    """Freeze the model and wrap every Linear whose name ends with a target. Returns the number wrapped."""
    model.requires_grad_(False)
    wrapped = 0
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                wrapped += 1
    if not wrapped:
        raise ValueError(f"no Linear layers named {targets} found for LoRA")
    return wrapped


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.state_dict().items() if ".lora_a." in k or ".lora_b." in k}


# ---- packing -------------------------------------------------------------------------------------------------------
@dataclass
class Packed:
    """One request as a sequence. qid: -1 state, >=0 question index. opt: -1 not an option, >=0 option index."""

    ids: list[int]
    pos: list[int]
    qid: list[int]
    opt: list[int]
    query: list[int]  # per question: index of its last token
    keys: list[list[int]]  # per question, per option: index of the option's last token


def _tok(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def pack(tokenizer, record: dict, max_state: int, max_branch: int) -> Packed:
    """Raises ValueError when a question with its options is longer than max_branch tokens (the state is truncated)."""
    ids = _tok(tokenizer, STATE_PREFIX + record["state"])[:max_state]
    if tokenizer.bos_token_id is not None and getattr(tokenizer, "add_bos_token", False):
        ids = [tokenizer.bos_token_id] + ids[: max_state - 1]
    n_state = len(ids)
    p = Packed(ids=list(ids), pos=list(range(n_state)), qid=[-1] * n_state, opt=[-1] * n_state, query=[], keys=[])
    for qi, q in enumerate(record["questions"]):
        q_ids = _tok(tokenizer, QUESTION_TEMPLATE.format(q["instr"]))
        opt_ids = [_tok(tokenizer, OPTION_TEMPLATE.format(o)) for o in q["options"]]
        if len(q_ids) + sum(len(o) for o in opt_ids) > max_branch:
            raise ValueError(f"question {qi} with its options exceeds {max_branch} tokens")
        start = len(p.ids)
        p.ids += q_ids
        p.pos += list(range(n_state, n_state + len(q_ids)))
        p.qid += [qi] * len(q_ids)
        p.opt += [-1] * len(q_ids)
        p.query.append(start + len(q_ids) - 1)
        option_start = n_state + len(q_ids)  # the same for every option: order invariance
        keys = []
        for oi, o in enumerate(opt_ids):
            p.ids += o
            p.pos += list(range(option_start, option_start + len(o)))
            p.qid += [qi] * len(o)
            p.opt += [oi] * len(o)
            keys.append(len(p.ids) - 1)
        p.keys.append(keys)
    return p


def attention_mask(qid: torch.Tensor, opt: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(B, L) segment ids -> additive (B, 1, L, L) mask. Padding (qid -2) attends only to itself."""
    length = qid.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=qid.device).tril()
    qi, qj = qid.unsqueeze(2), qid.unsqueeze(1)
    oi, oj = opt.unsqueeze(2), opt.unsqueeze(1)
    allow = causal & ((qj == -1) | ((qj == qi) & ((oj == -1) | (oj == oi))))
    allow = allow & (qi != -2) & (qj != -2)
    allow = allow | torch.eye(length, dtype=torch.bool, device=qid.device)
    return torch.zeros(allow.shape, dtype=dtype, device=qid.device).masked_fill(~allow, torch.finfo(dtype).min).unsqueeze(1)


def collate(packed: list[Packed], pad_id: int) -> dict[str, torch.Tensor]:
    length = max(len(p.ids) for p in packed)
    b = len(packed)
    ids = torch.full((b, length), pad_id, dtype=torch.long)
    pos = torch.zeros((b, length), dtype=torch.long)
    qid = torch.full((b, length), -2, dtype=torch.long)
    opt = torch.full((b, length), -1, dtype=torch.long)
    queries, keys = [], []
    for r, p in enumerate(packed):
        n = len(p.ids)
        ids[r, :n] = torch.tensor(p.ids)
        pos[r, :n] = torch.tensor(p.pos)
        qid[r, :n] = torch.tensor(p.qid)
        opt[r, :n] = torch.tensor(p.opt)
        queries += [r * length + i for i in p.query]
        keys += [[r * length + i for i in k] for k in p.keys]
    n_opts = max(len(k) for k in keys)
    key_idx = torch.full((len(keys), n_opts), -1, dtype=torch.long)
    for i, k in enumerate(keys):
        key_idx[i, : len(k)] = torch.tensor(k)
    return {"input_ids": ids, "position_ids": pos, "qid": qid, "opt": opt,
            "query_idx": torch.tensor(queries), "key_idx": key_idx}


# ---- model ---------------------------------------------------------------------------------------------------------
class DecoderDecider(nn.Module):
    def __init__(self, backbone: nn.Module, head_dim: int = 256, max_state: int = 2048, max_branch: int = 1024):
        super().__init__()
        self.backbone = backbone
        self.head = PointerHead(backbone.config.hidden_size, head_dim)
        self.max_state = max_state
        self.max_branch = max_branch

    def encode(self, tokenizer, records: list[dict], max_length: int | None = None):
        packed = [pack(tokenizer, r, self.max_state, self.max_branch) for r in records]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
        return collate(packed, pad), [len(p.ids) for p in packed]

    def forward(self, input_ids, position_ids, qid, opt, query_idx, key_idx) -> torch.Tensor:
        dtype = next(self.backbone.parameters()).dtype
        mask = attention_mask(qid, opt, dtype)
        hidden = self.backbone(input_ids=input_ids, attention_mask=mask, position_ids=position_ids).last_hidden_state
        return self.pointer(hidden.reshape(-1, hidden.shape[-1]), query_idx, key_idx)

    def pointer(self, flat_hidden: torch.Tensor, query_idx: torch.Tensor, key_idx: torch.Tensor) -> torch.Tensor:
        """Pointer head over flattened (rows * tokens, hidden) states: question-query tokens vs option-key tokens."""
        query = flat_hidden[query_idx].to(self.head.q.weight.dtype)
        keys = flat_hidden[key_idx.clamp(min=0)].to(self.head.q.weight.dtype)
        logits = self.head(query, keys).float()
        return logits.masked_fill(key_idx < 0, float("-inf"))

    # ---- state (document) cache ----------------------------------------------------------------------------------
    @torch.inference_mode()
    def logits_cached(self, tokenizer, record: dict, cache: StateCache) -> tuple[torch.Tensor, int]:
        """One record, reusing the state's keys/values when the same state was read before. Returns the same logits as
        forward() on this record (the state is causal and seen identically by every branch), plus the token count.

        The state runs once; its KV cache is kept (LRU). Each later request with that state only runs its question
        branches against the cached prefix."""
        p = pack(tokenizer, record, self.max_state, self.max_branch)
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype
        n_state = p.qid.count(-1)
        key = tuple(p.ids[:n_state])
        past = cache.get(key)
        if past is None:
            out = self.backbone(input_ids=torch.tensor([p.ids[:n_state]], device=device),
                                position_ids=torch.tensor([p.pos[:n_state]], device=device), use_cache=True)
            past = out.past_key_values
            cache.put(key, past)
        past = copy.deepcopy(past)  # the forward below appends to the cache in place
        qid = torch.tensor([p.qid[n_state:]], device=device)
        opt = torch.tensor([p.opt[n_state:]], device=device)
        branch = attention_mask(qid, opt, dtype)  # (1, 1, Lb, Lb): the branch rules among branch tokens
        mask = torch.cat([torch.zeros((1, 1, branch.shape[2], n_state), dtype=dtype, device=device), branch], dim=-1)
        hidden = self.backbone(input_ids=torch.tensor([p.ids[n_state:]], device=device), attention_mask=mask,
                               position_ids=torch.tensor([p.pos[n_state:]], device=device), past_key_values=past,
                               use_cache=True).last_hidden_state[0]
        n_opts = max(len(k) for k in p.keys)
        key_idx = torch.full((len(p.keys), n_opts), -1, dtype=torch.long, device=device)
        for i, k in enumerate(p.keys):
            key_idx[i, : len(k)] = torch.tensor([j - n_state for j in k], device=device)
        query = hidden[torch.tensor([j - n_state for j in p.query], device=device)].to(self.head.q.weight.dtype)
        keys = hidden[key_idx.clamp(min=0)].to(self.head.q.weight.dtype)
        logits = self.head(query, keys).float().masked_fill(key_idx < 0, float("-inf"))
        return logits, len(p.ids)


class StateCache:
    """LRU of state KV caches, bounded by entries and by total cached tokens (KV memory grows with tokens)."""

    def __init__(self, size: int = 32, max_tokens: int = 65536):
        self.size = size
        self.max_tokens = max_tokens
        self.entries: OrderedDict[tuple, object] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: tuple):
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.entries.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: tuple, value) -> None:
        if not self.size or len(key) > self.max_tokens:
            return
        self.entries[key] = value
        while len(self.entries) > self.size or sum(len(k) for k in self.entries) > self.max_tokens:
            self.entries.popitem(last=False)

    def stats(self) -> dict:
        return {"states": len(self.entries), "hits": self.hits, "misses": self.misses}


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """For inference: fold every LoRA adapter into its base weight (W + scale * B @ A) and drop the adapter. Same
    function, but one matmul per layer instead of three, and no extra kernels. Returns the number merged."""
    merged = 0
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                delta = (child.lora_b.weight.float() @ child.lora_a.weight.float()) * child.scale
                base.weight.copy_((base.weight.float() + delta).to(base.weight.dtype))
                setattr(module, name, base)
                merged += 1
    return merged
