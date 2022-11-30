import copy
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from .mha import MultiheadAttention
from .layernorm import LayerNorm
from .utils import transform_weight


def _get_activation_fn(activation):
    return activation


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        activation="relu",
        layer_norm_eps=1e-05,
        batch_first=False,
        norm_first=False,
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.norm_first = norm_first

    @classmethod
    def from_torch(cls, mod):
        transformer_encoder_layer = cls(
            mod.self_attn.embed_dim,
            mod.self_attn.num_heads,
            mod.linear1.out_features,
            mod.activation,
            mod.norm1.eps,
        )
        state_dict = transform_weight(mod)
        transformer_encoder_layer.load_state_dict(state_dict)

        return transformer_encoder_layer

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(
            src,
            src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + src2
        src = self.norm1(src)
        src2 = self.linear2((self.activation(self.linear1(src))))
        src = src + src2
        src = self.norm2(src)

        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2 = self.self_attn(
            src2,
            src2,
            value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src + src2
        src2 = self.norm2(src)
        src2 = self.linear2(self.activation(self.linear1(src2)))
        src = src + src2

        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.norm_first:
            return self.forward_pre(src, src_mask, src_key_padding_mask)
        return self.forward_post(src, src_mask, src_key_padding_mask)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.multihead_attn = MultiheadAttention(d_model, nhead)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    @classmethod
    def from_torch(cls, mod):
        transformer_decoder_layer = cls(
            mod.self_attn.embed_dim,
            mod.self_attn.num_heads,
            mod.linear1.out_features,
            mod.activation,
            mod.norm1.eps,
        )
        state_dict = transform_weight(mod)
        transformer_decoder_layer.load_state_dict(state_dict)

        return transformer_decoder_layer

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    @classmethod
    def from_torch(cls, mod):
        encoder_layer_pytorch = mod.layers[0]
        encoder_layer = TransformerEncoderLayer.from_torch(encoder_layer_pytorch)
        transformer_encoder = cls(encoder_layer, mod.num_layers, mod.norm)
        state_dict = transform_weight(mod)
        transformer_encoder.load_state_dict(state_dict)

        return transformer_encoder

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    @classmethod
    def from_torch(cls, mod):
        decoder_layer_pytorch = mod.layers[0]
        decoder_layer = TransformerDecoderLayer.from_torch(decoder_layer_pytorch)
        transformer_decoder = cls(decoder_layer, mod.num_layers, mod.norm)
        state_dict = transform_weight(mod)
        transformer_decoder.load_state_dict(state_dict)

        return transformer_decoder

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        activation="relu",
        custom_encoder=None,
        custom_decoder=None,
        layer_norm_eps=1e-5,
        batch_first=False,
    ):
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                activation,
                layer_norm_eps,
                batch_first,
            )
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                activation,
                layer_norm_eps,
                batch_first,
            )
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    @classmethod
    def from_torch(cls, mod):
        transformer = cls(
            mod.d_model,
            mod.nhead,
            mod.encoder.num_layers,
            mod.decoder.num_layers,
            mod.encoder.layers[0].linear1.out_features,
            mod.encoder.layers[0].activation,
            None,
            None,
            mod.encoder.norm.eps,
            mod.batch_first,
        )
        state_dict = transform_weight(mod)
        transformer.load_state_dict(state_dict)
        logger.info("replace torch transformer")

        return transformer

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None,
        memory_mask=None,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):

        memory = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
