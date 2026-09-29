"""Per question type calibration and cascade thresholds."""

import copy

import pytest

from dragonfly.engine import Cascade, parse_threshold

STATE = "the customer asks about card delivery"
NOUL = {"instr": "is this positive ?", "options": ["no", "yes"], "qtype": "noul"}
CHOICE = {"instr": "which intent", "options": ["card", "refund", "late"], "qtype": "choice"}


def test_per_type_temperature_applies_to_its_type_only(engine):
    base = engine.probs([{"state": STATE, "questions": [NOUL, CHOICE]}])[0][0]
    engine.config.temperature_by_type = {"noul": 50.0}
    hot = engine.probs([{"state": STATE, "questions": [NOUL, CHOICE]}])[0][0]
    assert abs(hot[0][0] - 0.5) < abs(base[0][0] - 0.5) + 1e-9  # the noul answer flattened towards 50/50
    assert hot[1] == pytest.approx(base[1], abs=1e-6)  # the choice answer untouched
    assert engine.temperature_for("score") == engine.config.temperature


def test_threshold_parsing():
    assert parse_threshold("0.7") == 0.7
    assert parse_threshold("noul=0.6, choice=0.7,default=0.5") == {"noul": 0.6, "choice": 0.7, "default": 0.5}


def test_cascade_escalates_by_type(engine):
    large = copy.deepcopy(engine)
    rec = {"state": STATE, "questions": [NOUL, CHOICE]}
    only_choice = Cascade(engine, large, {"noul": 0.0, "choice": 1.01})  # never escalate noul, always choice
    _, _, tiers = only_choice.probs([rec])
    assert tiers[0] == ["S", "S"] and only_choice.escalated == 1  # the tiny test engines are both tier S
    assert only_choice.threshold_for("score") == 0.8  # no default given
