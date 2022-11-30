import torch
import collections


def mha_update(state_dict, prefix=""):
    brocolli_state_dict = collections.OrderedDict()

    weight_name = ".".join(filter(None, [prefix, "in_proj_weight"]))
    bias_name = ".".join(filter(None, [prefix, "in_proj_bias"]))
    w_q, w_k, w_v = state_dict[weight_name].chunk(3)
    b_q, b_k, b_v = state_dict[bias_name].chunk(3)

    q_proj_weight_name = ".".join(filter(None, [prefix, "q_proj.weight"]))
    k_proj_weight_name = ".".join(filter(None, [prefix, "k_proj.weight"]))
    v_proj_weight_name = ".".join(filter(None, [prefix, "v_proj.weight"]))

    brocolli_state_dict[q_proj_weight_name] = w_q
    brocolli_state_dict[k_proj_weight_name] = w_k
    brocolli_state_dict[v_proj_weight_name] = w_v

    q_proj_bias_name = ".".join(filter(None, [prefix, "q_proj.bias"]))
    k_proj_bias_name = ".".join(filter(None, [prefix, "k_proj.bias"]))
    v_proj_bias_name = ".".join(filter(None, [prefix, "v_proj.bias"]))

    brocolli_state_dict[q_proj_bias_name] = b_q
    brocolli_state_dict[k_proj_bias_name] = b_k
    brocolli_state_dict[v_proj_bias_name] = b_v

    state_dict.pop(weight_name)
    state_dict.pop(bias_name)

    return brocolli_state_dict


def transform_mha_weight(state_dict):
    brocolli_state_dict = collections.OrderedDict()
    brocolli_state_dict.update(mha_update(state_dict))
    brocolli_state_dict.update(state_dict)

    return brocolli_state_dict


def transform_transformer_encoder_layer_weight(state_dict, prefix=""):
    brocolli_state_dict = collections.OrderedDict()
    prefix_ = ".".join(filter(None, [prefix, "self_attn"]))
    brocolli_state_dict.update(mha_update(state_dict, prefix=prefix_))
    brocolli_state_dict.update(state_dict)

    return brocolli_state_dict


def transform_transformer_decoder_layer_weight(state_dict, prefix=""):
    brocolli_state_dict = collections.OrderedDict()
    prefix_ = ".".join(filter(None, [prefix, "self_attn"]))
    brocolli_state_dict.update(mha_update(state_dict, prefix=prefix_))
    prefix_ = ".".join(filter(None, [prefix, "multihead_attn"]))
    brocolli_state_dict.update(mha_update(state_dict, prefix=prefix_))
    brocolli_state_dict.update(state_dict)

    return brocolli_state_dict


def transform_transformer_encoder_weight(state_dict, num_layers):
    brocolli_state_dict = collections.OrderedDict()

    for index in range(num_layers):
        prefix_ = "layers.{}.self_attn".format(index)
        brocolli_state_dict.update(mha_update(state_dict, prefix=prefix_))

    brocolli_state_dict.update(state_dict)

    return brocolli_state_dict


def transform_transformer_decoder_weight(state_dict, num_layers):
    brocolli_state_dict = collections.OrderedDict()

    for index in range(num_layers):
        prefix_ = "layers.{}.self_attn".format(index)
        brocolli_state_dict.update(mha_update(state_dict, prefix=prefix_))
        prefix_ = "layers.{}.multihead_attn".format(index)
        brocolli_state_dict.update(mha_update(state_dict, prefix=prefix_))

    brocolli_state_dict.update(state_dict)

    return brocolli_state_dict


def transform_transformer_weight(state_dict, num_encoder_layers, num_decoder_layers):
    brocolli_state_dict = collections.OrderedDict()

    for index in range(num_encoder_layers):
        prefix_ = "encoder.layers.{}.self_attn".format(index)
        brocolli_state_dict.update(mha_update(state_dict, prefix=prefix_))

    for index in range(num_decoder_layers):
        prefix_ = "decoder.layers.{}.self_attn".format(index)
        brocolli_state_dict.update(mha_update(state_dict, prefix=prefix_))
        prefix_ = "decoder.layers.{}.multihead_attn".format(index)
        brocolli_state_dict.update(mha_update(state_dict, prefix=prefix_))

    brocolli_state_dict.update(state_dict)

    return brocolli_state_dict


def transform_weight(module):
    if isinstance(module, torch.nn.TransformerEncoderLayer):
        return transform_transformer_encoder_layer_weight(module.state_dict())
    elif isinstance(module, torch.nn.TransformerDecoderLayer):
        return transform_transformer_decoder_layer_weight(module.state_dict())
    elif isinstance(module, torch.nn.MultiheadAttention):
        return transform_mha_weight(module.state_dict())
    elif isinstance(module, torch.nn.TransformerEncoder):
        return transform_transformer_encoder_weight(
            module.state_dict(), module.num_layers
        )
    elif isinstance(module, torch.nn.TransformerDecoder):
        return transform_transformer_decoder_weight(
            module.state_dict(), module.num_layers
        )
    elif isinstance(module, torch.nn.Transformer):
        return transform_transformer_weight(
            module.state_dict(), module.encoder.num_layers, module.decoder.num_layers
        )
    else:
        raise ValueError("Unknown module: {}".format(module))
