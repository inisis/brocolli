from loguru import logger

from .base_layer import BaseLayer
from .gemm_func import GemmFunc


class LinearFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(LinearFunc, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_node(self, name=None, params=None, attr_dict=None):
        if len(self._output_shape[0]) == 2:
            gemm_layer = GemmFunc(self._source_node, auto_gen=False)
            gemm_layer.add_bottom_top()
            if (
                "bias" in self._source_node.kwargs
                and self._source_node.kwargs["bias"] is not None
            ):
                bias_node = self._source_node.kwargs["bias"]
                gemm_layer._in_names.append(bias_node.name)
            gemm_layer.generate_node(self._source_node.name)
            self.node_post_process(gemm_layer)
