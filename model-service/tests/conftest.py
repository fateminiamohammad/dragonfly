"""A tiny offline tokenizer and random BERT backbone, so tests need no downloads and run on CPU in seconds."""

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from dragonfly.engine import CheckpointConfig, Engine, build_model
from dragonfly.models.encoder import EncoderDecider

WORDS = ("the a is was card payment arrived yes no refund delivery late good bad great terrible customer asks about "
         "which intent best describes this message how positive review rating one two three four five").split()


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {w: i for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", *WORDS, "-", ":", "?", ".", ","])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace()])
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                   cls_token="[CLS]", sep_token="[SEP]")


@pytest.fixture
def engine() -> Engine:
    torch.manual_seed(0)
    tok = tiny_tokenizer()
    backbone = BertModel(BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                    intermediate_size=64, max_position_embeddings=128), add_pooling_layer=False)
    return Engine(EncoderDecider(backbone, head_dim=16), tok, CheckpointConfig(backbone="tiny", max_length=128), "cpu")


def tiny_qwen():
    return Qwen3Model(Qwen3Config(vocab_size=len(tiny_tokenizer()), hidden_size=32, intermediate_size=64,
                                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                  max_position_embeddings=512))


@pytest.fixture
def decoder_engine() -> Engine:
    torch.manual_seed(0)
    config = CheckpointConfig(tier="M", backbone="tiny", head_dim=16, max_length=256, max_state=256, lora_r=4, lora_alpha=8)
    model = build_model(config, tiny_qwen())
    for name, p in model.named_parameters():  # LoRA B starts at zero; randomize so the adapters affect the output
        if "lora_b" in name:
            torch.nn.init.normal_(p, std=0.02)
    return Engine(model, tiny_tokenizer(), config, "cpu")
