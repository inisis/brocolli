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
            gemm_layer = GemmFunc(self._source_node)
            self.node_post_process(gemm_layer)
