from loguru import logger
import numpy as np
from onnx import helper
from onnx import TensorProto as tp

from .base_layer import BaseLayer


class ConvTransposeLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(ConvTransposeLayer, self).__init__(source_node, module, auto_gen)

    def get_conv_attr(self):
        attr_dict = {
            "dilations": [1, 1],  # list of ints defaults is 1
            "group": 1,  # int default is 1
            "kernel_shape": 1,  # list of ints If not present, should be inferred from input W.
            "pads": [0, 0, 0, 0],  # list of ints defaults to 0
            "strides": [1, 1],  # list of ints  defaults is 1
        }

        if isinstance(self._module.dilation, tuple):
            attr_dict["dilations"] = self._module.dilation
        else:
            attr_dict["dilations"] = [self._module.dilation]

        if isinstance(self._module.kernel_size, tuple):
            attr_dict["kernel_shape"] = self._module.kernel_size
        else:
            attr_dict["kernel_shape"] = [self._module.kernel_size]

        if isinstance(self._module.stride, tuple):
            attr_dict["strides"] = self._module.stride
        else:
            attr_dict["strides"] = [self._module.stride]

        if isinstance(self._module.padding, tuple):
            attr_dict["pads"] = self._module.padding * 2
        else:
            attr_dict["pads"] = [self._module.padding] * 4

        attr_dict["group"] = self._module.groups

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        self.create_params(self._name + "_weight", self._module.weight.detach().numpy())
        if self._module.bias is not None:
            self.create_params(self._name + "_bias", self._module.bias.detach().numpy())

        attr_dict = self.get_conv_attr()
        logger.debug(attr_dict)
        node = helper.make_node(
            "ConvTranspose", self._in_names, self._out_names, self._name, **attr_dict
        )
        logger.info("deconv_layer: " + self._name + " created")
        self._node.append(node)
