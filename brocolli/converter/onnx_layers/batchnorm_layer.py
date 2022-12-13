from loguru import logger
from onnx import helper
from onnx import TensorProto as tp


from .base_layer import BaseLayer


class BatchNormLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(BatchNormLayer, self).__init__(source_node, module, auto_gen)

    def get_batchnorm_attr(self):
        attr_dict = {"epsilon": 1e-5, "momentum": 0.999}

        attr_dict["epsilon"] = self._module.eps
        attr_dict["momentum"] = self._module.momentum

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        self.create_params(self._name + "_scale", self._module.weight.detach().numpy())
        self.create_params(self._name + "_bias", self._module.bias.detach().numpy())
        self.create_params(
            self._name + "_mean", self._module.running_mean.detach().numpy()
        )
        self.create_params(
            self._name + "_var", self._module.running_var.detach().numpy()
        )

        attr_dict = self.get_batchnorm_attr()
        logger.debug(attr_dict)

        node = helper.make_node(
            "BatchNormalization",
            self._in_names,
            self._out_names,
            self._name,
            **attr_dict
        )
        logger.info("batchnorm_layer: " + self._name + " created")

        self._node.append(node)
