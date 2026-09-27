import pytest
import torch

from dragonfly.calibration import auto_rate, brier, ece, fit_temperature


def test_ece_zero_when_confidence_matches_accuracy():
    conf = [0.75] * 4
    correct = [True, True, True, False]
    assert ece(conf, correct) == pytest.approx(0)


def test_ece_detects_overconfidence():
    assert ece([0.99] * 10, [True] * 5 + [False] * 5) == pytest.approx(0.49)


def test_brier_perfect_and_worst():
    assert brier([[1.0, 0.0]], [0]) == 0
    assert brier([[0.0, 1.0]], [0]) == 2


def test_auto_rate():
    conf = [0.99, 0.98, 0.97, 0.5]
    assert auto_rate(conf, [True, True, True, False], 0.05) == 0.75
    assert auto_rate(conf, [False, True, True, True], 0.05) == 0


def test_fit_temperature_recovers_scale():
    torch.manual_seed(0)
    true_logits = torch.randn(4000, 4) * 2
    labels = torch.distributions.Categorical(logits=true_logits).sample().tolist()
    overconfident = [row * 3 for row in true_logits]  # a model 3x too sharp
    assert fit_temperature(overconfident, labels) == pytest.approx(3, rel=0.1)
