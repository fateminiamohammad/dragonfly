import pytest
import torch

from dragonfly.models.graphs import OPTIONS, ROWS, TOKENS, bucket


def test_bucket_rounds_up_and_rejects_oversized():
    assert bucket(1, ROWS) == 1 and bucket(3, ROWS) == 4 and bucket(64, ROWS) == 64
    assert bucket(65, ROWS) is None
    assert bucket(102, TOKENS) == 128 and bucket(5000, TOKENS) is None
    assert bucket(77, OPTIONS) == 128


def test_cpu_engine_ignores_cuda_graphs(engine):
    engine.enable_cuda_graphs()
    assert engine.graphs is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_graph_replay_matches_eager(engine):
    """Padding to buckets must not change results (fp32, so only padding can cause a difference)."""
    engine.model.cuda()
    engine.device, engine.autocast = "cuda", False
    records = [{"state": "the customer asks about card delivery",
                "questions": [{"instr": "which intent", "options": ["card arrived", "refund", "payment late"]},
                              {"instr": "is the customer good ?", "options": ["no", "yes"]}]}]
    eager, _, _ = engine.probs(records)
    engine.enable_cuda_graphs()
    graphed, _, _ = engine.probs(records)
    assert engine.graphs.stats()["replays"] == 1
    for a, b in zip(eager[0], graphed[0]):
        assert a == pytest.approx(b, abs=1e-5)
