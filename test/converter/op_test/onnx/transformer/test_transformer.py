import torch
import torch.nn as nn
import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


def test_mha_basic(
    shape=[1, 3],
):
    from brocolli.converter.pytorch_layer.mha import MultiheadAttention

    batch_size, seq_len, feature_dim, head_num = 7, 12, 16, 4
    model = MultiheadAttention(feature_dim, head_num)
    shape = (
        (batch_size, seq_len, feature_dim),
        (batch_size, seq_len, feature_dim),
        (batch_size, seq_len, feature_dim),
    )
    concrete_args = {"key_padding_mask": None, "need_weights": False, "attn_mask": None}
    Tester("mha_basic", model, shape, concrete_args=concrete_args)


def test_layernorm_basic(
    shape=[1, 3],
):
    from brocolli.converter.pytorch_layer.layernorm import LayerNorm

    batch, sentence_length, embedding_dim = 20, 5, 10
    model = LayerNorm(embedding_dim)
    shape = (batch, sentence_length, embedding_dim)
    Tester("layernorm_basic", model, shape)


def test_transformer_encoder_layer_basic(
    shape=(32, 10, 512),
):
    from brocolli.converter.pytorch_layer.transformer import TransformerEncoderLayer

    model = TransformerEncoderLayer(d_model=512, nhead=8, batch_first=False)
    shape = (32, 10, 512)
    concrete_args = {"src_mask": None, "src_key_padding_mask": None}
    Tester("transformer_encoder_layer_basic", model, shape, concrete_args=concrete_args)


def test_transformer_decoder_layer_basic(
    shape=(32, 10, 512),
):
    from brocolli.converter.pytorch_layer.transformer import TransformerDecoderLayer

    model = TransformerDecoderLayer(d_model=512, nhead=8)
    shape = ((10, 32, 512), (20, 32, 512))
    concrete_args = {
        "tgt_mask": None,
        "memory_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
    }
    Tester("transformer_decoder_layer_basic", model, shape, concrete_args=concrete_args)


def test_transformer_encoder_basic(
    shape=(32, 10, 512),
):
    from brocolli.converter.pytorch_layer.transformer import (
        TransformerEncoderLayer,
        TransformerEncoder,
    )

    encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
    model = TransformerEncoder(encoder_layer, num_layers=2)
    shape = (10, 32, 512)
    concrete_args = {"mask": None, "src_key_padding_mask": None}
    Tester("transformer_encoder_basic", model, shape, concrete_args=concrete_args)


def test_transformer_decoder_basic(
    shape=(32, 10, 512),
):
    from brocolli.converter.pytorch_layer.transformer import (
        TransformerDecoderLayer,
        TransformerDecoder,
    )

    decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8)
    model = TransformerDecoder(decoder_layer, num_layers=2)
    shape = ((10, 32, 512), (20, 32, 512))
    concrete_args = {
        "tgt_mask": None,
        "memory_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
    }
    Tester("transformer_decoder_basic", model, shape, concrete_args=concrete_args)


def test_transformer_basic(
    shape=(32, 10, 512),
):
    from brocolli.converter.pytorch_layer.transformer import Transformer

    model = Transformer(nhead=16, num_encoder_layers=2, num_decoder_layers=2)
    shape = (10, 32, 512), (20, 32, 512)
    concrete_args = {
        "src_mask": None,
        "tgt_mask": None,
        "memory_mask": None,
        "src_key_padding_mask": None,
        "tgt_key_padding_mask": None,
        "memory_key_padding_mask": None,
    }
    Tester("transformer_basic", model, shape, concrete_args=concrete_args)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        [
            "-p",
            "no:warnings",
            "-v",
            "test/converter/op_test/onnx/transformer/test_transformer.py",
        ]
    )
