from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

from .base_layer import BaseLayer
from .slice_func import SliceFunc
from .concat_func import ConcatFunc
from .squeeze_func import SqueezeFunc
from .reshape_func import ReshapeFunc
from .permute_func import PermuteFunc
from .transpose_func import TransposeFunc


class LSTMLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        self.num_layers = module.num_layers
        self.num_direction = 2 if module.bidirectional else 1
        super(LSTMLayer, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def gen_lstm_block_outnames(self, idx=0):
        if self.num_layers == 1:
            if self._module.batch_first:
                out_names = [self._name + "_layer_0_out"]
            else:
                out_names = [self._name + "_0"]
            if len(self._output_shape) != 1:
                out_names.extend([self._name + "_1", self._name + "_2"])
        else:
            if idx == self.num_layers - 1:
                if self._module.batch_first:
                    out_names = [
                        self._name + "_layer_" + str(self.num_layers - 1) + "_out"
                    ]
                else:
                    out_names = [self._name + "_0"]
                if len(self._output_shape) != 1:
                    out_names.extend(
                        [
                            self._name + "_layer_" + str(idx) + "_out_last_hidden",
                            self._name + "_layer_" + str(idx) + "_out_last_cell",
                        ]
                    )
            else:
                out_names = [self._name + "_layer_" + str(idx) + "_out"]
                if len(self._output_shape) != 1:
                    out_names.extend(
                        [
                            self._name + "_layer_" + str(idx) + "_out_last_hidden",
                            self._name + "_layer_" + str(idx) + "_out_last_cell",
                        ]
                    )

        return out_names

    def generate_node(self, name=None, params=None, attr_dict=None):
        if self._module.num_layers == 1:
            if self._module.batch_first:
                transpose_layer = TransposeFunc(self._source_node, auto_gen=False)
                transpose_layer.add_bottom_top(
                    in_names=[self.recursive_find_name(self._source_node.args[0])],
                    out_names=[self._name + "_input_transpose"],
                )
                transpose_layer.generate_node(
                    name=self._name + "_input_transpose", attr_dict={"perm": [1, 0, 2]}
                )
                self.node_post_process(transpose_layer)
            lstm_block = LSTMBlock(self._source_node, self._module)
            if self._module.batch_first:
                in_names = [self._name + "_input_transpose"]
            else:
                in_names = [self.recursive_find_name(self._source_node.args[0])]
            if len(self._input_shape) != 1:
                in_names.extend(
                    [
                        self.recursive_find_name(self._source_node.args[1]) + "_0",
                        self.recursive_find_name(self._source_node.args[1]) + "_1",
                    ]
                )
            out_names = self.gen_lstm_block_outnames()
            lstm_block.generate_block(0, in_names=in_names, out_names=out_names)
            self.node_post_process(lstm_block)
        else:
            if self._module.batch_first:
                transpose_layer = TransposeFunc(self._source_node, auto_gen=False)
                transpose_layer.add_bottom_top(
                    in_names=[self.recursive_find_name(self._source_node.args[0])],
                    out_names=[self._name + "_input_transpose"],
                )
                transpose_layer.generate_node(
                    name=self._name + "_input_transpose", attr_dict={"perm": [1, 0, 2]}
                )
                self.node_post_process(transpose_layer)
            lstm_block = LSTMBlock(self._source_node, self._module)
            if self._module.batch_first:
                in_names = [self._name + "_input_transpose"]
            else:
                in_names = [self.recursive_find_name(self._source_node.args[0])]
            if len(self._input_shape) != 1:
                in_names.extend(
                    [
                        self.recursive_find_name(self._source_node.args[1]) + "_0",
                        self.recursive_find_name(self._source_node.args[1]) + "_1",
                    ]
                )
            out_names = self.gen_lstm_block_outnames(0)
            lstm_block.generate_block(0, in_names=in_names, out_names=out_names)
            self.node_post_process(lstm_block)

            for idx in range(1, self._module.num_layers - 1):
                lstm_block = LSTMBlock(self._source_node, self._module)
                if len(self._input_shape) == 1:
                    in_names = [
                        self._source_node.name + "_layer_" + str(idx - 1) + "_out"
                    ]
                else:
                    in_names = [
                        self._source_node.name + "_layer_" + str(idx - 1) + "_out",
                        self.recursive_find_name(self._source_node.args[1]) + "_0",
                        self.recursive_find_name(self._source_node.args[1]) + "_1",
                    ]
                out_names = self.gen_lstm_block_outnames(idx)
                lstm_block.generate_block(idx, in_names=in_names, out_names=out_names)
                self.node_post_process(lstm_block)

            lstm_block = LSTMBlock(self._source_node, self._module)
            if len(self._input_shape) == 1:
                in_names = [
                    self._source_node.name
                    + "_layer_"
                    + str(self.num_layers - 2)
                    + "_out"
                ]
            else:
                in_names = [
                    self._source_node.name
                    + "_layer_"
                    + str(self.num_layers - 2)
                    + "_out",
                    self.recursive_find_name(self._source_node.args[1]) + "_0",
                    self.recursive_find_name(self._source_node.args[1]) + "_1",
                ]

            out_names = self.gen_lstm_block_outnames(self.num_layers - 1)
            lstm_block.generate_block(
                self._module.num_layers - 1, in_names=in_names, out_names=out_names
            )
            self.node_post_process(lstm_block)

            concat_layer = ConcatFunc(self._source_node, auto_gen=False)
            concat_layer.add_bottom_top(
                in_names=[
                    self._name + "_layer_" + str(idx) + "_out_last_hidden"
                    for idx in range(self.num_layers)
                ],
                out_names=[self._source_node.name + "_1"],
            )
            concat_layer.generate_node(
                name=self._source_node.name + "_hidden_concat", attr_dict={"axis": 0}
            )
            self.node_post_process(concat_layer)

            # add concat but if not marked as output, it will be removed later
            concat_layer = ConcatFunc(self._source_node, auto_gen=False)
            concat_layer.add_bottom_top(
                in_names=[
                    self._name + "_layer_" + str(idx) + "_out_last_cell"
                    for idx in range(self.num_layers)
                ],
                out_names=[self._source_node.name + "_2"],
            )
            concat_layer.generate_node(
                name=self._source_node.name + "_cell_concat", attr_dict={"axis": 0}
            )
            self.node_post_process(concat_layer)

        if self._module.batch_first:
            transpose_layer = TransposeFunc(self._source_node, auto_gen=False)
            transpose_layer.add_bottom_top(
                in_names=[self._name + "_layer_" + str(self.num_layers - 1) + "_out"],
                out_names=[self._name + "_0"],
            )
            transpose_layer.generate_node(
                name=self._name + "_output_transpose", attr_dict={"perm": [1, 0, 2]}
            )
            self.node_post_process(transpose_layer)


class LSTMBlock(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(LSTMBlock, self).__init__(source_node, module, auto_gen)
        self.num_layers = module.num_layers
        self.num_direction = 2 if module.bidirectional else 1

    def generate_node(self, name=None, params=None, attr_dict=None):
        pass

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def get_permute_weight(self, weight):
        weight = weight.detach().numpy()
        # pytorch is input, forget, cell, output.
        # onnx is    input, output, forget, cell.
        params = np.array_split(weight, 4, axis=0)
        params = np.concatenate((params[0], params[3], params[1], params[2]), axis=0)
        return params

    def get_lstm_params(self, layer_index=0):
        params = [None] * 4
        weight_ih_name = "weight_ih_l{0}".format(layer_index)
        weight_hh_name = "weight_hh_l{0}".format(layer_index)
        weight_ih_l0, weight_hh_l0 = getattr(self._module, weight_ih_name), getattr(
            self._module, weight_hh_name
        )
        if self._module.bidirectional is False:
            params[0], params[1] = (
                self.get_permute_weight(weight_ih_l0)[None, ...],
                self.get_permute_weight(weight_hh_l0)[None, ...],
            )
        else:
            weight_ih_reverse_name = "weight_ih_l{0}_reverse".format(layer_index)
            weight_hh_reverse_name = "weight_hh_l{0}_reverse".format(layer_index)
            weight_ih_l0_reverse, weight_hh_l0_reverse = (
                getattr(self._module, weight_ih_reverse_name),
                getattr(self._module, weight_hh_reverse_name),
            )
            params[0], params[1] = np.stack(
                [
                    self.get_permute_weight(weight_ih_l0),
                    self.get_permute_weight(weight_ih_l0_reverse),
                ],
                axis=0,
            ), np.stack(
                [
                    self.get_permute_weight(weight_hh_l0),
                    self.get_permute_weight(weight_hh_l0_reverse),
                ],
                axis=0,
            )

        if self._module.bias is not False:
            bias_ih_name = "bias_ih_l{0}".format(layer_index)
            bias_hh_name = "bias_hh_l{0}".format(layer_index)
            bias_ih_l0, bias_hh_l0 = getattr(self._module, bias_ih_name), getattr(
                self._module, bias_hh_name
            )
            if self._module.bidirectional is False:
                params[2] = np.concatenate(
                    [
                        self.get_permute_weight(bias_ih_l0)[None, ...],
                        self.get_permute_weight(bias_hh_l0)[None, ...],
                    ],
                    axis=1,
                )
            else:
                bias_ih_reverse_name = "bias_ih_l{0}_reverse".format(layer_index)
                bias_hh_reverse_name = "bias_hh_l{0}_reverse".format(layer_index)
                bias_ih_l0_reverse, bias_hh_l0_reverse = (
                    getattr(self._module, bias_ih_reverse_name),
                    getattr(self._module, bias_hh_reverse_name),
                )
                params[2] = np.concatenate(
                    [
                        np.stack(
                            [
                                self.get_permute_weight(bias_ih_l0),
                                self.get_permute_weight(bias_ih_l0_reverse),
                            ],
                            axis=0,
                        ),
                        np.stack(
                            [
                                self.get_permute_weight(bias_hh_l0),
                                self.get_permute_weight(bias_hh_l0_reverse),
                            ],
                            axis=0,
                        ),
                    ],
                    axis=1,
                )

        return params

    def generate_slice(self, name, block_id, in_names, out_names):
        slice_layer = SliceFunc(self._source_node, auto_gen=False)
        slice_layer.add_bottom_top(in_names=in_names, out_names=out_names)
        params_slice = [
            np.array([block_id * self.num_direction]),
            np.array([(block_id + 1) * self.num_direction]),
            np.array([0]),
            np.array([1]),
        ]
        slice_layer.generate_node(name, params_slice)
        self.node_post_process(slice_layer)

    def generate_block(self, block_id, in_names, out_names):
        if len(self._input_shape) != 1:
            self.generate_slice(
                self._source_node.name + "_" + str(block_id) + "_hidden_slice",
                block_id,
                in_names=[in_names[1]],
                out_names=[
                    self._source_node.name + "_" + str(block_id) + "_hidden_slice"
                ],
            )
            self.generate_slice(
                self._source_node.name + "_" + str(block_id) + "_cell_slice",
                block_id,
                in_names=[in_names[2]],
                out_names=[
                    self._source_node.name + "_" + str(block_id) + "_cell_slice"
                ],
            )

        lstm_cell = LSTMCell(self._source_node, self._module, auto_gen=False)
        if len(self._output_shape) != 1:
            lstm_cell.add_bottom_top(
                in_names=[in_names[0]],
                out_names=[
                    self._source_node.name + "_" + str(block_id) + "_out",
                    out_names[1],
                    out_names[2],
                ],
            )
        else:
            lstm_cell.add_bottom_top(
                in_names=[in_names[0]],
                out_names=[self._source_node.name + "_" + str(block_id) + "_out"],
            )

        params = self.get_lstm_params(block_id)

        lstm_cell.generate_params(params, self._source_node.name + "_" + str(block_id))
        if len(self._input_shape) != 1:
            lstm_cell._in_names.append(
                self._source_node.name + "_" + str(block_id) + "_hidden_slice"
            )
            lstm_cell._in_names.append(
                self._source_node.name + "_" + str(block_id) + "_cell_slice"
            )
        lstm_cell.generate_node(self._source_node.name + "_" + str(block_id))
        self.node_post_process(lstm_cell)

        if self._module.bidirectional is False:
            squeeze_layer = SqueezeFunc(self._source_node, auto_gen=False)
            squeeze_layer.add_bottom_top(
                in_names=[self._source_node.name + "_" + str(block_id) + "_out"],
                out_names=[out_names[0]],
            )
            squeeze_layer.generate_node(
                name=self._source_node.name + "_" + str(block_id) + "_squeeze",
                params=np.array([1]),
            )
            self.node_post_process(squeeze_layer)
        else:
            permute_layer = PermuteFunc(self._source_node, auto_gen=False)
            permute_layer.add_bottom_top(
                in_names=[self._source_node.name + "_" + str(block_id) + "_out"],
                out_names=[self._source_node.name + "_" + str(block_id) + "_permute"],
            )
            permute_layer.generate_node(
                name=self._source_node.name + "_" + str(block_id) + "_permute",
                attr_dict={"perm": [0, 2, 1, 3]},
            )
            self.node_post_process(permute_layer)

            reshape_layer = ReshapeFunc(self._source_node, auto_gen=False)
            reshape_layer.add_bottom_top(
                in_names=[self._source_node.name + "_" + str(block_id) + "_permute"],
                out_names=[out_names[0]],
            )
            reshape_layer.generate_node(
                name=self._source_node.name + "_" + str(block_id) + "_reshape",
                params=np.array([0, 0, -1]),
            )
            self.node_post_process(reshape_layer)


class LSTMCell(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(LSTMCell, self).__init__(source_node, module, auto_gen)

    def get_lstm_attr(self):

        attr_dict = {
            "hidden_size": [1],  # list of ints defaults is 1
        }

        attr_dict["hidden_size"] = self._module.hidden_size
        attr_dict["layout"] = 0  # 1 cannot infer
        if self._module.bidirectional is not False:
            attr_dict["direction"] = "bidirectional"

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        attr_dict = self.get_lstm_attr()
        logger.debug(attr_dict)
        node = helper.make_node(
            "LSTM", self._in_names, self._out_names, self._name, **attr_dict
        )
        logger.info("lstm_layer: " + self._name + " created")
        self._node.append(node)

    def generate_params(self, params, name=None):
        if name is not None:
            self._name = name

        self.create_params(self._name + "_weight", params[0])
        self.create_params(self._name + "_recurrent_weight", params[1])
        self.create_params(self._name + "_bias", params[2])
        self.create_params(self._name + "_sequence_lens ", params[3])
