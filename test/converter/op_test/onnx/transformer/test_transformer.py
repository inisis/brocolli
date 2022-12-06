import torch
import torch.nn as nn
import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


class TestTransformerClass:
    def test_mha(self, request):
        from brocolli.converter.pytorch_layer.mha import MultiheadAttention

        class MultiheadAttentionModel(nn.Module):
            def __init__(self):
                super(MultiheadAttentionModel, self).__init__()
                self.mha = MultiheadAttention(16, 4)

            def forward(self, q, k, v):
                return self.mha(q, k, v)

        batch_size, seq_len, feature_dim, head_num = 7, 12, 16, 4
        model = MultiheadAttentionModel()
        shape = (
            (batch_size, seq_len, feature_dim),
            (batch_size, seq_len, feature_dim),
            (batch_size, seq_len, feature_dim),
        )

        q = torch.rand(shape[0])
        k = torch.rand(shape[1])
        v = torch.rand(shape[2])
        Tester(request.node.name, model, (q, k, v))

    def test_layernorm(self, request):
        from brocolli.converter.pytorch_layer.layernorm import LayerNorm

        batch, sentence_length, embedding_dim = 20, 5, 10
        model = LayerNorm(embedding_dim)
        shape = (batch, sentence_length, embedding_dim)
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_transformer_encoder_layer(self, request):
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
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_transformer_decoder_layer(self, request):
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
        tgt = torch.rand(shape[0])
        memory = torch.rand(shape[1])
        Tester(request.node.name, model, (tgt, memory))

    def test_transformer_encoder(self, request):
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
        x = torch.rand(shape)
        Tester(request.node.name, model, x)

    def test_transformer_decoder(self, request):
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
        tgt = torch.rand(shape[0])
        memory = torch.rand(shape[1])
        Tester(request.node.name, model, (tgt, memory))

    def test_transformer(self, request):
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
        src = torch.rand(shape[0])
        tgt = torch.rand(shape[1])
        Tester(request.node.name, model, (src, tgt))


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
