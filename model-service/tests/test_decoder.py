"""Dragonfly-M guarantees: order invariance, question isolation, packing = separate passes, padding-safe batching."""

import pytest
import torch

from dragonfly.engine import Engine
from dragonfly.models.decoder import LoRALinear, attention_mask, pack

STATE = "the customer asks about card delivery . the payment was late"
Q1 = {"instr": "which intent best describes this message", "options": ["card arrived", "refund", "payment late", "good"]}
Q2 = {"instr": "is the customer positive ?", "options": ["no", "yes"]}


def probs(engine, *records):
    return engine.probs(list(records))[0]


def test_option_order_cannot_change_scores(decoder_engine):
    base = probs(decoder_engine, {"state": STATE, "questions": [Q1]})[0][0]
    perm = [2, 0, 3, 1]
    shuffled = {**Q1, "options": [Q1["options"][i] for i in perm]}
    got = probs(decoder_engine, {"state": STATE, "questions": [shuffled]})[0][0]
    assert got == pytest.approx([base[i] for i in perm], abs=1e-5)


def test_questions_cannot_see_each_other(decoder_engine):
    alone = probs(decoder_engine, {"state": STATE, "questions": [Q1]})[0][0]
    other = {"instr": "rating", "options": ["one", "two", "three", "four", "five"]}
    for neighbour in (Q2, other):
        packed = probs(decoder_engine, {"state": STATE, "questions": [Q1, neighbour]})[0][0]
        assert packed == pytest.approx(alone, abs=1e-5)


def test_second_question_same_packed_or_alone(decoder_engine):
    alone = probs(decoder_engine, {"state": STATE, "questions": [Q2]})[0][0]
    packed = probs(decoder_engine, {"state": STATE, "questions": [Q1, Q2]})[0][1]
    assert packed == pytest.approx(alone, abs=1e-5)


def test_batch_padding_does_not_change_answers(decoder_engine):
    short = {"state": "a good review", "questions": [Q2]}
    long = {"state": STATE + " " + STATE, "questions": [Q1, Q2]}
    alone = probs(decoder_engine, short)[0]
    batched = probs(decoder_engine, short, long)[0]
    assert batched[0] == pytest.approx(alone[0], abs=1e-5)


def test_mask_isolates_segments():
    # state(2) | q0 (1) | q0 opt0 (1) | q0 opt1 (1) | pad
    qid = torch.tensor([[-1, -1, 0, 0, 0, -2]])
    opt = torch.tensor([[-1, -1, -1, 0, 1, -1]])
    allowed = attention_mask(qid, opt, torch.float32)[0, 0] == 0
    assert allowed[4].tolist() == [True, True, True, False, True, False]  # opt1 sees state, question, itself; not opt0
    assert allowed[5].tolist() == [False] * 5 + [True]  # padding sees only itself


def test_options_share_start_position(decoder_engine):
    p = pack(decoder_engine.tokenizer, {"state": STATE, "questions": [Q1]}, 256, 256)
    starts = [p.pos[p.opt.index(k)] for k in range(len(Q1["options"]))]
    assert len(set(starts)) == 1


def test_too_long_branch_is_rejected(decoder_engine):
    big = {"instr": "which", "options": ["the card " * 40] * 10}
    with pytest.raises(ValueError):
        pack(decoder_engine.tokenizer, {"state": STATE, "questions": [big]}, 256, 128)


def test_only_lora_and_head_train(decoder_engine):
    names = [n for n, p in decoder_engine.model.named_parameters() if p.requires_grad]
    assert names and all(".lora_" in n or n.startswith("head.") for n in names)
    assert any(isinstance(m, LoRALinear) for m in decoder_engine.model.modules())


def test_save_and_load_roundtrip(decoder_engine, tmp_path):
    base_dir = tmp_path / "base"
    fresh = type(decoder_engine.model.backbone)(decoder_engine.model.backbone.config)
    # the saved base must be the frozen weights the adapters were trained on: copy them over first
    fresh.load_state_dict({k.replace(".base.", "."): v for k, v in decoder_engine.model.backbone.state_dict().items()
                           if ".lora_" not in k})
    fresh.save_pretrained(base_dir)
    decoder_engine.config.backbone = str(base_dir)
    decoder_engine.save(tmp_path / "ck")
    assert (tmp_path / "ck" / "lora.pt").exists() and not (tmp_path / "ck" / "backbone").exists()
    loaded = Engine.load(str(tmp_path / "ck"), "cpu")
    rec = {"state": STATE, "questions": [Q1, Q2]}
    assert probs(loaded, rec)[0][0] == pytest.approx(probs(decoder_engine, rec)[0][0], abs=1e-5)
