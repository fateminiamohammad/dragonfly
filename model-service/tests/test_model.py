import pytest
import torch

from dragonfly.engine import Engine
from dragonfly.models.encoder import encode_rows, question_text

ROWS = [
    {"instr": "which intent", "options": ["card arrived", "refund", "payment late"]},
    {"instr": "is the customer good ?", "options": ["no", "yes"]},
]
STATES = ["the customer asks about card delivery", "a great review"]


def test_question_text_spans_cover_options():
    text, spans = question_text("q", ["alpha", "beta"])
    assert [text[s:e] for s, e in spans] == ["alpha", "beta"]


def test_option_mask_marks_only_option_tokens(engine):
    batch = encode_rows(engine.tokenizer, ROWS, STATES, 128)
    assert batch["option_mask"].shape[:2] == (2, 3)
    ids = batch["input_ids"]
    words = [engine.tokenizer.convert_ids_to_tokens(ids[0][batch["option_mask"][0, k]].tolist()) for k in range(3)]
    assert words == [["card", "arrived"], ["refund"], ["payment", "late"]]
    assert not batch["option_mask"][1, 2].any()  # row 2 has only two options


def test_probs_sum_to_one_and_respect_option_count(engine):
    probs, tokens, tiers = engine.probs([{"state": STATES[0], "questions": ROWS}])
    assert [len(p) for p in probs[0]] == [3, 2]
    for p in probs[0]:
        assert sum(p) == pytest.approx(1, abs=1e-5)
    assert tokens[0] > 0
    assert tiers == [["S", "S"]]


def test_truncation_cuts_state_not_options(engine):
    long_state = " ".join(["the"] * 500)
    batch = encode_rows(engine.tokenizer, ROWS[:1], [long_state], 32)
    assert batch["input_ids"].shape[1] == 32
    assert batch["option_mask"][0].any(-1).all()


def test_empty_option_is_rejected(engine):
    with pytest.raises(ValueError):
        encode_rows(engine.tokenizer, [{"instr": "q", "options": ["yes", ""]}], ["the"], 64)


def test_save_and_load_roundtrip(engine, tmp_path):
    engine.config.temperature = 1.7
    engine.save(tmp_path / "ck")
    loaded = Engine.load(str(tmp_path / "ck"), "cpu")
    assert loaded.config.temperature == 1.7
    rec = [{"state": STATES[0], "questions": ROWS}]
    a = engine.probs(rec)[0]
    b = loaded.probs(rec)[0]
    assert torch.allclose(torch.tensor(a[0][0]), torch.tensor(b[0][0]), atol=1e-5)


def test_head_norm_bounds_logits_and_old_checkpoints_still_load(engine, tmp_path):
    from dragonfly.models.encoder import PointerHead
    torch.manual_seed(0)
    q, k = torch.randn(2, 32) * 1000, torch.randn(2, 3, 32) * 1000  # huge activations, like a bigger backbone
    plain, normed = PointerHead(32, 16), PointerHead(32, 16, norm=True)
    normed.load_state_dict(plain.state_dict(), strict=False)
    assert plain(q, k).abs().max() > 100 * normed(q, k).abs().max()
    engine.save(tmp_path / "old")  # saved without head_norm (the default): loads exactly as before
    assert not Engine.load(str(tmp_path / "old"), "cpu").config.head_norm
