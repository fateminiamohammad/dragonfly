"""Image -> tags with SigLIP-2: non-autoregressive. One image forward pass gives an embedding; one matrix multiply scores
it against every tag in the vocabulary (whose text embeddings are computed once at startup).

Each tag gets two numbers:
  score  the tag's share of the image among all tags (softmax over the vocabulary). Relative, sums to 1; use it to rank.
  p      SigLIP's own sigmoid probability. Independent per tag, but NOT calibrated for a large generic vocabulary: a
         correct "cat" can score 0.003 while text-heavy images score 0.9. Do not threshold on it.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from .device import gpu_dtype

log = logging.getLogger(__name__)
DEFAULT_TAGS = Path(__file__).with_name("tags_default.txt")
PROMPT = "a photo of {}."


def decode_image(data: bytes, max_pixels: int = 40_000_000) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > max_pixels:
            raise ValueError(f"image is {img.width}x{img.height}; too large")
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"could not decode image: {e}") from e


def load_tags(path: str | Path | None) -> list[str]:
    lines = Path(path or DEFAULT_TAGS).read_text(encoding="utf-8").splitlines()
    tags = [t.strip() for t in lines if t.strip() and not t.startswith("#")]
    return list(dict.fromkeys(tags))


class Tagger:
    def __init__(self, model_id: str, device: str, tags: list[str], model=None, processor=None):
        from transformers import AutoModel, AutoProcessor

        self.model_id = model_id
        self.device = device
        self.tags = tags
        self.processor = processor or AutoProcessor.from_pretrained(model_id)
        dtype = gpu_dtype(device)
        self.model = (model or AutoModel.from_pretrained(model_id, dtype=dtype)).to(device).eval()
        self.text = self._embed_tags(tags)
        # CUDA graphs per batch-size bucket: the image tower's input is always (n, 3, H, W) at the model's fixed size,
        # so one capture per bucket replays ~all its kernels in one launch (19 ms -> a few ms measured on an RTX 3090 Ti)
        self.graphs: dict[int, tuple] = {}
        self.use_graphs = device == "cuda"

    def _features(self, out) -> torch.Tensor:
        # get_*_features returns a tensor in older transformers and an output object with pooler_output in newer ones
        return out if isinstance(out, torch.Tensor) else out.pooler_output

    @torch.inference_mode()
    def _embed_tags(self, tags: list[str]) -> torch.Tensor:
        chunks = []
        for i in range(0, len(tags), 256):
            batch = self.processor(text=[PROMPT.format(t) for t in tags[i:i + 256]], padding="max_length",
                                   max_length=64, truncation=True, return_tensors="pt")
            feats = self._features(self.model.get_text_features(**{k: v.to(self.device) for k, v in batch.items()}))
            chunks.append(torch.nn.functional.normalize(feats.float(), dim=-1))
        log.info("embedded %d tags with %s", len(tags), self.model_id)
        return torch.cat(chunks)

    def _graphed(self, pixels: torch.Tensor) -> torch.Tensor:
        n = pixels.shape[0]
        size = next((b for b in (1, 2, 4, 8, 16, 32) if b >= n), None)
        if size is None:
            return self._features(self.model.get_image_features(pixel_values=pixels))
        if size not in self.graphs:
            static = torch.zeros((size, *pixels.shape[1:]), dtype=pixels.dtype, device=pixels.device)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.model.get_image_features(pixel_values=static)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._features(self.model.get_image_features(pixel_values=static))
            self.graphs[size] = (graph, static, out)
            log.info("captured CUDA graph for %d image(s)", size)
        graph, static, out = self.graphs[size]
        static.zero_()
        static[:n].copy_(pixels)
        graph.replay()
        return out[:n].clone()

    @torch.inference_mode()
    def tag(self, images: list[Image.Image], top_k: int = 8, min_score: float = 0.02) -> list[list[dict]]:
        """Top tags per image by score (at least one, at most top_k, dropping those with score < min_score)."""
        inputs = self.processor(images=images, return_tensors="pt")
        dtype = next(self.model.parameters()).dtype
        pixels = inputs["pixel_values"].to(self.device, dtype)
        if self.use_graphs and set(inputs) == {"pixel_values"}:
            feats = self._graphed(pixels)
        else:
            feats = self._features(self.model.get_image_features(pixel_values=pixels))
        img = torch.nn.functional.normalize(feats.float(), dim=-1)
        scale = self.model.logit_scale.float().exp()
        bias = self.model.logit_bias.float()
        logits = img @ self.text.T * scale + bias
        shares = torch.softmax(logits, dim=-1).cpu().numpy()
        probs = torch.sigmoid(logits).cpu().numpy()
        out = []
        for share, prob in zip(shares, probs):
            order = np.argsort(-share)[:top_k]
            out.append([{"label": self.tags[i], "score": round(float(share[i]), 4), "p": round(float(prob[i]), 4)}
                        for n, i in enumerate(order) if n == 0 or share[i] >= min_score])
        return out
