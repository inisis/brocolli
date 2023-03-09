import pytest
import warnings


def test_glu():
    import torch
    import torch.nn as nn
    from brocolli.converter.pytorch_layer.glu import GLU

    batch_size, seq_len, feature_dim, head_num = 7, 12, 16, 4
    x = torch.rand((batch_size, seq_len, feature_dim))
    model_pytorch = nn.GLU(dim=-1)
    model_pytorch.eval()
    out_pytorch = model_pytorch(x)

    model_brocolli = GLU(dim=-1)
    model_brocolli.eval()

    out_brocolli = model_brocolli(x)

    tol = 1e-5
    torch.testing.assert_close(out_pytorch, out_brocolli, rtol=tol, atol=tol)


def test_mha():
    import torch
    import torch.nn as nn
    from brocolli.converter.pytorch_layer.mha import MultiheadAttention
    from brocolli.converter.pytorch_layer.utils import transform_weight

    batch_size, seq_len, feature_dim, head_num = 7, 12, 16, 4
    q = torch.rand((batch_size, seq_len, feature_dim))
    k = torch.rand((batch_size, seq_len, feature_dim))
    v = torch.rand((batch_size, seq_len, feature_dim))

    model_pytorch = nn.MultiheadAttention(feature_dim, head_num)
    model_pytorch.eval()
    out_pytorch = model_pytorch(q, k, v)[0]

    state_dict = transform_weight(model_pytorch)
    model_brocolli = MultiheadAttention(feature_dim, head_num)
    model_brocolli.load_state_dict(state_dict)
    model_brocolli.eval()

    out_brocolli = model_brocolli(q, k, v)[0]

    tol = 1e-5
    torch.testing.assert_close(out_pytorch, out_brocolli, rtol=tol, atol=tol)


@pytest.mark.parametrize(("normalized_shape"), [([10]), ([10, 10]), ([5, 10, 10])])
@pytest.mark.parametrize(("elementwise_affine"), (True, False))
def test_layernorm(normalized_shape, elementwise_affine):
    import torch
    import torch.nn as nn
    from brocolli.converter.pytorch_layer.layernorm import LayerNorm

    N, C, H, W = 20, 5, 10, 10
    embedding = torch.randn(N, C, H, W)
    model_pytorch = nn.LayerNorm(
        normalized_shape, elementwise_affine=elementwise_affine
    )
    model_pytorch.eval()

    out_pytorch = model_pytorch(embedding)

    model_brocolli = LayerNorm.from_torch(model_pytorch)
    model_brocolli.eval()

    out_brocolli = model_brocolli(embedding)

    tol = 1e-5
    torch.testing.assert_close(out_pytorch, out_brocolli, rtol=tol, atol=tol)


def test_transformer_encoder_layer():
    import torch
    import torch.nn as nn
    from brocolli.converter.pytorch_layer.transformer import TransformerEncoderLayer
    from brocolli.converter.pytorch_layer.utils import transform_weight

    model_pytorch = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=False)
    model_pytorch.eval()

    src = torch.rand(32, 10, 512)
    out_pytorch = model_pytorch(src)

    model_brocolli = TransformerEncoderLayer.from_torch(model_pytorch)
    model_brocolli.eval()

    out_brocolli = model_brocolli(src)

    tol = 1e-5
    torch.testing.assert_close(out_pytorch, out_brocolli, rtol=tol, atol=tol)


def test_transformer_decoder_layer():
    import torch
    import torch.nn as nn
    from brocolli.converter.pytorch_layer.transformer import TransformerDecoderLayer

    model_pytorch = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=False)
    model_pytorch.eval()

    memory = torch.rand(10, 32, 512)
    tgt = torch.rand(20, 32, 512)
    out_pytorch = model_pytorch(tgt, memory)

    model_brocolli = TransformerDecoderLayer.from_torch(model_pytorch)
    model_brocolli.eval()

    out_brocolli = model_brocolli(tgt, memory)

    tol = 1e-5
    torch.testing.assert_close(out_pytorch, out_brocolli, rtol=tol, atol=tol)


def test_transformer_encoder():
    import torch
    import torch.nn as nn
    from brocolli.converter.pytorch_layer.transformer import TransformerEncoder

    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    model_pytorch = nn.TransformerEncoder(encoder_layer, num_layers=2)
    model_pytorch.eval()

    src = torch.rand(10, 32, 512)
    out_pytorch = model_pytorch(src)

    model_brocolli = TransformerEncoder.from_torch(model_pytorch)

    model_brocolli.eval()

    out_brocolli = model_brocolli(src)

    tol = 1e-5
    torch.testing.assert_close(out_pytorch, out_brocolli, rtol=tol, atol=tol)


def test_transformer_decoder():
    import torch
    import torch.nn as nn
    from brocolli.converter.pytorch_layer.transformer import TransformerDecoder

    decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    model_pytorch = nn.TransformerDecoder(decoder_layer, num_layers=2)
    model_pytorch.eval()

    memory = torch.rand(10, 32, 512)
    tgt = torch.rand(20, 32, 512)
    out_pytorch = model_pytorch(tgt, memory)

    model_brocolli = TransformerDecoder.from_torch(model_pytorch)

    model_brocolli.eval()

    out_brocolli = model_brocolli(tgt, memory)

    tol = 1e-5
    torch.testing.assert_close(out_pytorch, out_brocolli, rtol=tol, atol=tol)


def test_transformer():
    import torch
    import torch.nn as nn
    from brocolli.converter.pytorch_layer.transformer import Transformer

    model_pytorch = nn.Transformer(nhead=16, num_encoder_layers=2, num_decoder_layers=2)
    model_pytorch.eval()

    src = torch.rand((10, 32, 512))
    tgt = torch.rand(20, 32, 512)
    out_pytorch = model_pytorch(src, tgt)

    model_brocolli = Transformer.from_torch(model_pytorch)
    model_brocolli.eval()

    out_brocolli = model_brocolli(src, tgt)

    tol = 1e-5
    torch.testing.assert_close(out_pytorch, out_brocolli, rtol=tol, atol=tol)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(["-p", "no:warnings", "-v", "test/converter/test_pytorch_layer.py"])
