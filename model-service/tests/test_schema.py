import pytest
from pydantic import ValidationError

from dragonfly.schema import (
    DecideRequest,
    choice_confidence,
    render,
    score_confidence,
    to_answers,
    to_record,
)

REQUEST = {
    "state": {"message": "my card arrived late", "customer": {"tier": "gold"}},
    "questions": {
        "intent": {"type": "choice", "instructions": "Which intent?", "criteria": {"card_arrival": "card delivery",
                                                                                  "refund": None}},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        "sentiment": {"type": "score", "instructions": "How positive?", "criteria": ["bad", "ok", "great"]},
    },
}


def test_to_record_maps_every_type_to_options():
    record, meta = to_record(DecideRequest.model_validate(REQUEST))
    assert "message: my card arrived late" in record["state"] and "tier: gold" in record["state"]
    options = [q["options"] for q in record["questions"]]
    assert options == [["card_arrival: card delivery", "refund"], ["no", "yes"], ["bad", "ok", "great"]]
    assert [m["keys"] for m in meta] == [["card_arrival", "refund"], ["false", "true"], ["0", "1", "2"]]


def test_to_answers_shapes():
    _, meta = to_record(DecideRequest.model_validate(REQUEST))
    answers = to_answers([[0.9, 0.1], [0.3, 0.7], [0.0, 0.5, 0.5]], meta)
    assert answers["intent"]["choice"] == "card_arrival"
    assert answers["urgent"]["noul"] == 0.7
    assert answers["sentiment"]["score"] == 1.5
    assert answers["sentiment"]["legend"] == {"0": "bad", "1": "ok", "2": "great"}


def test_confidence_bounds():
    assert choice_confidence([0.5, 0.5]) == 0
    assert choice_confidence([1.0, 0.0, 0.0]) == 1
    assert score_confidence([0, 0, 1]) == 1
    assert score_confidence([1 / 3] * 3) == pytest.approx(0, abs=1e-9)


def test_option_limits():
    # a choice may have up to 10,000 options (answered in two stages above 255); a score stays within one pass
    DecideRequest.model_validate({"state": "x", "questions": {"q": {"type": "choice",
                                                                   "criteria": {str(i): None for i in range(256)}}}})
    bad = {"state": "x", "questions": {"q": {"type": "choice", "criteria": {str(i): None for i in range(10001)}}}}
    with pytest.raises(ValidationError):
        DecideRequest.model_validate(bad)
    with pytest.raises(ValidationError):
        DecideRequest.model_validate({"state": "x", "questions": {"q": {"type": "score", "criteria": list(range(256))}}})


def test_render_lists_and_scalars():
    assert render(["a", {"b": 1}]) == "- a\n- b: 1"
    assert render(None) == ""
