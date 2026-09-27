"""Calibration metrics and temperature scaling. A calibrated model's "70% sure" is right about 70% of the time."""

from __future__ import annotations

import math

import torch


def ece(confidence: list[float], correct: list[bool], bins: int = 10) -> float:
    """Expected calibration error: bin by confidence, average |accuracy - confidence| weighted by bin size."""
    n = len(confidence)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidence) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confidence[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return total


def brier(probs: list[list[float]], labels: list[int]) -> float:
    """Mean squared distance between the predicted distribution and the one-hot label (0 = perfect)."""
    if not probs:
        return 0.0
    return sum(sum((p - (i == y)) ** 2 for i, p in enumerate(row)) for row, y in zip(probs, labels)) / len(probs)


def nll(probs: list[list[float]], labels: list[int]) -> float:
    return -sum(math.log(max(row[y], 1e-12)) for row, y in zip(probs, labels)) / max(len(probs), 1)


def auto_rate(confidence: list[float], correct: list[bool], error_budget: float = 0.05) -> float:
    """Share of decisions that can be automated while keeping their error rate within the budget: accept the most
    confident answers first and stop where the accepted error rate would exceed it."""
    order = sorted(range(len(confidence)), key=lambda i: -confidence[i])
    best, wrong = 0, 0
    for n, i in enumerate(order, 1):
        wrong += not correct[i]
        if wrong / n <= error_budget:
            best = n
    return best / max(len(confidence), 1)


def pad_logits(logits: list[torch.Tensor]) -> torch.Tensor:
    """Questions have different option counts; pad to (N, K_max) with -inf so softmax ignores the padding."""
    k = max(len(x) for x in logits)
    out = torch.full((len(logits), k), float("-inf"))
    for i, x in enumerate(logits):
        out[i, : len(x)] = x
    return out


def fit_temperature(logits: list[torch.Tensor], labels: list[int]) -> float:
    """The temperature T minimizing the NLL of softmax(logits / T) on held-out data."""
    z = pad_logits(logits).float()
    y = torch.tensor(labels)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))
