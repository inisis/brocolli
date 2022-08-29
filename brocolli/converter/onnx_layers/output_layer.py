from loguru import logger
from onnx import helper
from onnx import TensorProto as tp
from torch.fx.passes.shape_prop import TensorMetadata

from onnx_layers.base_layer import BaseLayer


class OutputLayer(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(OutputLayer, self).__init__(source_node, module, auto_gen)
        if self._auto_gen:
            output_name = self.recursive_find_name(self._source_node.args[0])
            self.generate_output(output_name)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def get_shape_list(self, tensor_meta):
        shape_list = []
        for tensor in tensor_meta:
            if isinstance(tensor, TensorMetadata):
                shape_list.append(tensor.shape)
            else:
                shape_list.extend(self.get_shape_list(tensor))

        return shape_list

    def generate_output(self, name):
        if isinstance(self.tensor_meta, TensorMetadata):
            output_tvi = helper.make_tensor_value_info(
                name, tp.FLOAT, self.tensor_meta.shape
            )
            logger.info("output_layer: " + name + " created")
            self._out_tensor_value_info.append(output_tvi)
        else:
            shape_list = self.get_shape_list(self.tensor_meta)
            for idx, shape in enumerate(shape_list):
                output_tvi = helper.make_tensor_value_info(
                    name + "_" + str(idx), tp.FLOAT, shape
                )
                logger.info(
                    "output_layer: "
                    + self._source_node.args[0].name
                    + "_"
                    + str(idx)
                    + " created"
                )
                self._out_tensor_value_info.append(output_tvi)
