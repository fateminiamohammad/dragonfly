"""CUDA graphs for the tier S forward pass.

Profiling showed an eager forward costs ~7 ms of GPU work but ~40 ms of CPU time: about 680 small kernels, each
launched separately (launches are especially slow on Windows/WSL2). A CUDA graph records the whole forward once and
replays it with one launch.

Graphs need fixed shapes, so inputs are padded up to buckets of (rows, tokens, options). Each bucket is captured the
first time it is needed (a one-off cost of a few hundred ms) and replayed afterwards. Shapes beyond the largest bucket
fall back to the eager path. Padding cannot change the answers: padded rows are discarded, padded tokens are masked,
and padded options have no tokens so their logits are -inf and trimmed.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch

log = logging.getLogger(__name__)

ROWS = (1, 2, 4, 8, 16, 32, 64)
TOKENS = (64, 128, 256, 512, 1024, 1536)
OPTIONS = (2, 4, 8, 16, 32, 64, 128, 256)


def bucket(value: int, choices: tuple[int, ...]) -> int | None:
    return next((c for c in choices if c >= value), None)


class GraphRunner:
    def __init__(self, model: torch.nn.Module, max_graphs: int = 96):
        self.model = model
        self.max_graphs = max_graphs
        self.graphs: OrderedDict[tuple[int, int, int], tuple[torch.cuda.CUDAGraph, dict, torch.Tensor]] = OrderedDict()
        self.pool = None
        self.replays = 0
        self.fallbacks = 0

    def stats(self) -> dict:
        return {"captured": len(self.graphs), "replays": self.replays, "fallbacks": self.fallbacks}

    @torch.inference_mode()
    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Logits for a tier S batch already on the GPU, or None when the shape has no bucket (caller runs eager)."""
        rows, length = batch["input_ids"].shape
        options = batch["option_mask"].shape[1]
        key = (bucket(rows, ROWS), bucket(length, TOKENS), bucket(options, OPTIONS))
        if None in key:
            self.fallbacks += 1
            return None
        entry = self.graphs.get(key)
        if entry is None:
            entry = self._capture(key)
        else:
            self.graphs.move_to_end(key)
        graph, static, out = entry
        static["input_ids"].zero_()
        static["attention_mask"].zero_()
        static["attention_mask"][:, 0] = 1  # padded rows attend to one token, so they stay finite
        static["option_mask"].zero_()
        static["input_ids"][:rows, :length].copy_(batch["input_ids"])
        static["attention_mask"][:rows, :length].copy_(batch["attention_mask"])
        static["option_mask"][:rows, :options, :length].copy_(batch["option_mask"])
        graph.replay()
        self.replays += 1
        return out[:rows, :options].clone()

    def _capture(self, key: tuple[int, int, int]):
        rows, length, options = key
        device = next(self.model.parameters()).device
        static = {
            "input_ids": torch.zeros(rows, length, dtype=torch.long, device=device),
            "attention_mask": torch.ones(rows, length, dtype=torch.long, device=device),
            "option_mask": torch.zeros(rows, options, length, dtype=torch.bool, device=device),
        }
        static["option_mask"][:, :, 0] = True  # any valid content: the capture only records the kernels
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):  # warm up (allocator, cuBLAS handles) off the capture
                self.model(**static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self.model(**static)
        self.pool = graph.pool()
        if len(self.graphs) >= self.max_graphs:
            self.graphs.popitem(last=False)
        self.graphs[key] = (graph, static, out)
        log.info("captured CUDA graph for rows=%d tokens=%d options=%d (%d graphs)", rows, length, options, len(self.graphs))
        return self.graphs[key]


DECODER_ROWS = (1, 2, 4, 8, 16, 32, 64)  # the cascade can send a large batch of escalated requests
DECODER_TOKENS = (64, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048)  # fine: padding is real GPU work


class DecoderGraphRunner:
    """CUDA graphs for the tier M backbone (Qwen3 + LoRA): one eager forward launched about 3,100 kernels (36 ms of GPU
    work, about 115 ms wall on an RTX 3090 Ti). The packed batch is padded to a (rows, tokens) bucket; padding tokens get
    segment id -2, so by the attention mask they see only themselves and nothing real sees them. The pointer head runs
    outside the graph on the real rows and tokens."""

    def __init__(self, model: torch.nn.Module, max_graphs: int = 32):
        self.model = model
        self.max_graphs = max_graphs
        self.graphs: OrderedDict[tuple[int, int], tuple] = OrderedDict()
        self.pool = None
        self.replays = 0
        self.fallbacks = 0

    def stats(self) -> dict:
        return {"captured": len(self.graphs), "replays": self.replays, "fallbacks": self.fallbacks}

    @torch.inference_mode()
    def __call__(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
        from .decoder import attention_mask

        rows, length = batch["input_ids"].shape
        key = (bucket(rows, DECODER_ROWS), bucket(length, DECODER_TOKENS))
        if None in key:
            self.fallbacks += 1
            return None
        entry = self.graphs.get(key) or self._capture(key)
        self.graphs.move_to_end(key)
        graph, static, hidden = entry
        device = batch["qid"].device
        qid = torch.full(key, -2, dtype=torch.long, device=device)
        opt = torch.full(key, -1, dtype=torch.long, device=device)
        qid[:rows, :length] = batch["qid"]
        opt[:rows, :length] = batch["opt"]
        static["input_ids"].zero_()
        static["position_ids"].zero_()
        static["input_ids"][:rows, :length].copy_(batch["input_ids"])
        static["position_ids"][:rows, :length].copy_(batch["position_ids"])
        static["attention_mask"].copy_(attention_mask(qid, opt, static["attention_mask"].dtype))
        graph.replay()
        self.replays += 1
        flat = hidden[:rows, :length].reshape(-1, hidden.shape[-1])
        return self.model.pointer(flat, batch["query_idx"], batch["key_idx"])

    def _capture(self, key: tuple[int, int]):
        from .decoder import attention_mask

        device = next(self.model.parameters()).device
        dtype = next(self.model.backbone.parameters()).dtype
        qid = torch.full(key, -2, dtype=torch.long, device=device)
        static = {
            "input_ids": torch.zeros(key, dtype=torch.long, device=device),
            "position_ids": torch.zeros(key, dtype=torch.long, device=device),
            "attention_mask": attention_mask(qid, torch.full(key, -1, dtype=torch.long, device=device), dtype),
        }
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.model.backbone(**static)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            hidden = self.model.backbone(**static).last_hidden_state
        self.pool = graph.pool()
        if len(self.graphs) >= self.max_graphs:
            self.graphs.popitem(last=False)
        self.graphs[key] = (graph, static, hidden)
        log.info("captured tier M CUDA graph for rows=%d tokens=%d (%d graphs)", key[0], key[1], len(self.graphs))
        return self.graphs[key]
