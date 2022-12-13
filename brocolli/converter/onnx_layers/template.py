import numpy as np
from loguru import logger

from onnx import helper
from onnx import TensorProto as tp

from .base_layer import BaseLayer


class Template(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(Template, self).__init__(source_node, module, auto_gen)

    def get_template_attr(self):
        attr_dict = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1}

        return attr_dict

    def generate_node(self, name=None, params=None, attr_dict=None):
        if name is not None:
            self._name = name

        if params is None:
            # handle non user-defined situation optional
            params = np.array([self._source_node.args[1]])

        if attr_dict is None:
            # handle non user-defined situation optional
            attr_dict = self.get_template_attr()

        self.create_params(self._name, params)
        node = helper.make_node(
            "template", self._in_names, self._out_names, self._name, **attr_dict
        )
        logger.info("template_layer: " + self._name + " created")
        self._node.append(node)
