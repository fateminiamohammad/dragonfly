"""Choice questions with more than 255 options (product catalogs, routing to thousands of teams): two stages.

Stage 1 (prune) splits the options into chunks that fit one pass and scores every chunk in one request. A softmax
over one chunk is not comparable with another chunk's, so every chunk also carries the same anchor option (the first
one): log p(option) - log p(anchor) puts every option on one scale. With isolated options (tier M, packed tier S),
an option's score does not depend on the rest of its chunk, so this equals scoring all options at once.

Stage 2 (decide) asks the question again with only the top candidates (default 32), through the normal pipeline:
cascade, plugins, calibration. The answer's probabilities cover the finalists; `pruned_mass` is stage 1's estimate of
the probability left in the options it dropped.
"""

from __future__ import annotations

import math

from .schema import MAX_OPTIONS, MAX_OPTIONS_TWO_STAGE  # noqa: F401 (re-exported)

FINALISTS = 32


def chunks(keys: list[str], size: int = MAX_OPTIONS - 1) -> list[list[str]]:
    """Option keys per stage-1 question: the anchor (keys[0]) first, then up to `size` others."""
    anchor, rest = keys[0], keys[1:]
    return [[anchor, *rest[i:i + size]] for i in range(0, len(rest), size)] or [[anchor]]


def global_scores(chunk_keys: list[list[str]], chunk_probs: list[list[float]]) -> dict[str, float]:
    """Stage-1 probabilities per chunk -> one log-score per option, relative to the anchor (anchor = 0)."""
    scores: dict[str, float] = {}
    for keys, probs in zip(chunk_keys, chunk_probs):
        anchor = math.log(max(probs[0], 1e-30))
        for k, p in zip(keys, probs):
            scores.setdefault(k, math.log(max(p, 1e-30)) - anchor)
    return scores


def finalists(scores: dict[str, float], k: int = FINALISTS) -> tuple[list[str], float]:
    """The k best options (in their original order is not needed: best first) and the pruned probability mass."""
    best = sorted(scores, key=scores.get, reverse=True)
    top, rest = best[:k], best[k:]
    peak = scores[best[0]]
    total = sum(math.exp(scores[x] - peak) for x in best)
    pruned = sum(math.exp(scores[x] - peak) for x in rest) / total if rest else 0.0
    return top, pruned
