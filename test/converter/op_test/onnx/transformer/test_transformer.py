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
    class TransformerEncoderLayerModel(nn.Module):
        def __init__(
            self,
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        ):
            super(TransformerEncoderLayerModel, self).__init__()
            self.transformer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            self.linear = nn.Linear(d_model, 1)

        def forward(self, src):
            output = self.transformer(src)
            output = self.linear(output)

            return output

    model = TransformerEncoderLayerModel(d_model=512, nhead=8)
    shape = (32, 10, 512)
    Tester("transformer_encoder_layer_basic", model, shape)


def test_transformer_decoder_layer_basic(
    shape=(32, 10, 512),
):
    class TransformerDecoderLayerModel(nn.Module):
        def __init__(
            self,
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        ):
            super(TransformerDecoderLayerModel, self).__init__()
            self.transformer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            self.linear = nn.Linear(d_model, 1)

        def forward(self, tgt, memory):
            output = self.transformer(tgt, memory)
            output = self.linear(output)

            return output

    model = TransformerDecoderLayerModel(d_model=512, nhead=8)
    shape = ((10, 32, 512), (20, 32, 512))
    Tester("transformer_decoder_layer_basic", model, shape)


def test_transformer_encoder_basic(
    shape=(32, 10, 512),
):
    class TransformerEncoderModel(nn.Module):
        def __init__(
            self,
            num_layers=6,
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        ):
            super(TransformerEncoderModel, self).__init__()
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                ),
                num_layers=num_layers,
            )
            self.linear = nn.Linear(d_model, 1)

        def forward(self, src):
            output = self.transformer(src)
            output = self.linear(output)

            return output

    model = TransformerEncoderModel(num_layers=2)
    shape = (10, 32, 512)
    Tester("transformer_encoder_basic", model, shape)


def test_transformer_decoder_basic(
    shape=(32, 10, 512),
):
    class TransformerDecoderModel(nn.Module):
        def __init__(
            self,
            num_layers=6,
            d_model=512,
            nhead=8,            
        ):
            super(TransformerDecoderModel, self).__init__()
            self.transformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                ),
                num_layers=num_layers,
            )
            self.linear = nn.Linear(d_model, 1)

        def forward(self, tgt, memory):
            output = self.transformer(tgt, memory)
            output = self.linear(output)

            return output

    model = TransformerDecoderModel(num_layers=2)
    shape = ((10, 32, 512), (20, 32, 512))
    Tester("transformer_decoder_basic", model, shape)


def test_transformer_basic(
    shape=(32, 10, 512),
):

    class TransformerModel(nn.Module):
        def __init__(
            self,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
        ):
            super(TransformerModel, self).__init__()
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            self.linear = nn.Linear(d_model, 1)

        def forward(self, src, tgt):
            output = self.transformer(src, tgt)
            output = self.linear(output)

            return output

    model = TransformerModel(nhead=16, num_encoder_layers=2, num_decoder_layers=2)
    shape = (10, 32, 512), (20, 32, 512)
    Tester("transformer_basic", model, shape)


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
