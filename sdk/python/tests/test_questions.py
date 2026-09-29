import enum
from typing import Literal

import pytest
from dragonfly_client import Decision, Scale, question


class Team(enum.Enum):
    BILLING = "billing"
    SHIPPING = "shipping"


def test_types_become_questions():
    assert question(bool) == {"type": "noul"}
    assert question(Literal["a", "b"], "which?") == {"type": "choice", "criteria": {"a": None, "b": None},
                                                     "instructions": "which?"}
    assert question(Team)["criteria"] == {"billing": None, "shipping": None}
    assert question(["x", "y"])["criteria"] == {"x": None, "y": None}
    assert question({"x": "the x one"})["criteria"] == {"x": "the x one"}
    assert question(Scale(["low", "high"])) == {"type": "score", "criteria": ["low", "high"]}
    assert question({"type": "noul", "instructions": "raw"}) == {"type": "noul", "instructions": "raw"}
    with pytest.raises(TypeError):
        question(int)


def test_decisions_read_every_answer_shape():
    yes = Decision.from_answer({"type": "noul", "noul": 0.8, "confidence": 0.6})
    assert yes and yes.value is True and yes.probabilities["true"] == 0.8
    assert Decision.from_answer({"type": "noul", "probability": 0.2, "confidence": 0.6}).value is False  # Kev's shape
    c = Decision.from_answer({"type": "choice", "choice": "b", "confidence": 0.5, "probabilities": {"a": 0.2, "b": 0.8}})
    assert c.value == "b"
    with pytest.raises(TypeError):
        bool(c)
