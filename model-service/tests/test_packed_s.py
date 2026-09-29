"""Packed tier S: one sequence per request with a segment mask. The state is read once; questions and options cannot
see each other, so packing, question order and option order never change a score."""

import pytest
import torch
from conftest import tiny_tokenizer
from transformers import ModernBertConfig, ModernBertModel

from dragonfly.engine import CheckpointConfig, Engine, build_model
from dragonfly.models.encoder import pack_encoder, packed_masks

STATE = "the customer asks about card delivery . the payment was late"
Q1 = {"instr": "which intent best describes this message", "options": ["card arrived", "refund", "payment late", "good"],
      "qtype": "choice"}
Q2 = {"instr": "is the customer positive ?", "options": ["no", "yes"], "qtype": "noul"}


@pytest.fixture
def packed_engine() -> Engine:
    torch.manual_seed(0)
    tok = tiny_tokenizer()
    backbone = ModernBertModel(ModernBertConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2,
        global_attn_every_n_layers=3, local_attention=4, max_position_embeddings=512, pad_token_id=0,
        bos_token_id=2, eos_token_id=3, cls_token_id=2, sep_token_id=3, attn_implementation="sdpa"))
    config = CheckpointConfig(tier="S", backbone="tiny", head_dim=16, max_length=256, packed=True, head_norm=True)
    return Engine(build_model(config, backbone), tok, config, "cpu")


def probs(engine, *records):
    return engine.probs(list(records))[0]


def test_packing_equals_asking_each_question_alone(packed_engine):
    together = probs(packed_engine, {"state": STATE, "questions": [Q1, Q2]})[0]
    alone = [probs(packed_engine, {"state": STATE, "questions": [q]})[0][0] for q in (Q1, Q2)]
    for got, want in zip(together, alone):
        assert got == pytest.approx(want, abs=1e-5)


def test_question_and_option_order_cannot_change_scores(packed_engine):
    base = probs(packed_engine, {"state": STATE, "questions": [Q1, Q2]})[0]
    swapped = probs(packed_engine, {"state": STATE, "questions": [Q2, Q1]})[0]
    assert swapped[0] == pytest.approx(base[1], abs=1e-5) and swapped[1] == pytest.approx(base[0], abs=1e-5)
    perm = [2, 0, 3, 1]
    shuffled = {**Q1, "options": [Q1["options"][i] for i in perm]}
    got = probs(packed_engine, {"state": STATE, "questions": [shuffled]})[0][0]
    assert got == pytest.approx([base[0][i] for i in perm], abs=1e-5)


def test_batching_and_padding_do_not_change_answers(packed_engine):
    short = {"state": "a good review", "questions": [Q2]}
    long = {"state": STATE + " " + STATE, "questions": [Q1, Q2]}
    alone = probs(packed_engine, short)[0]
    assert probs(packed_engine, short, long)[0][0] == pytest.approx(alone[0], abs=1e-5)


def test_the_state_is_read_once():
    tok = tiny_tokenizer()
    one = pack_encoder(tok, {"state": STATE, "questions": [Q1]}, 512)
    p = pack_encoder(tok, {"state": STATE, "questions": [Q1, Q2, Q1]}, 512)
    n_state = p["qid"].count(-1)
    assert n_state == len(tok(STATE).input_ids)  # [CLS] state [SEP], once
    q1 = len(one["ids"]) - n_state
    assert len(p["ids"]) == n_state + 2 * q1 + p["qid"].count(1)  # more questions add only their own tokens
    starts = [p["pos"][i] for i in p["query"]]
    assert starts == [n_state] * 3  # every question block starts at the same position


def test_masks_isolate_segments_and_respect_the_window():
    # state(3) | q0 text(1) | q0 opt0(1) | q0 opt1(1) | pad
    qid = torch.tensor([[-1, -1, -1, 0, 0, 0, -2]])
    opt = torch.tensor([[-1, -1, -1, -1, 0, 1, -1]])
    pos = torch.tensor([[0, 1, 2, 3, 4, 4, 0]])
    m = packed_masks(qid, opt, pos, window=1)
    full, local = m["full_attention"][0, 0], m["sliding_attention"][0, 0]
    assert not full[0, 3] and not full[0, 4]  # the state never sees questions or options
    assert full[4, :4].all() and not full[4, 5]  # an option sees state + question, not the other option
    assert not full[3, 4]  # the question text does not see its options
    assert full[6].tolist() == [False] * 6 + [True]  # padding sees only itself
    assert not local[4, 0] and local[4, 3]  # the local window follows positions


def test_questions_that_do_not_fit_are_rejected():
    tok = tiny_tokenizer()
    with pytest.raises(ValueError, match="alone need"):
        pack_encoder(tok, {"state": STATE, "questions": [Q1] * 10}, 32)


def test_packed_checkpoint_roundtrip(packed_engine, tmp_path):
    before = probs(packed_engine, {"state": STATE, "questions": [Q1, Q2]})
    packed_engine.save(str(tmp_path))
    loaded = Engine.load(str(tmp_path), "cpu")
    assert loaded.config.packed
    after = probs(loaded, {"state": STATE, "questions": [Q1, Q2]})
    assert after[0][0] == pytest.approx(before[0][0], abs=1e-5)
