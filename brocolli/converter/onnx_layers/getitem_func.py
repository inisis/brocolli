import sys
from loguru import logger
import numpy as np

from .base_layer import BaseLayer
from .slice_func import SliceFunc


class GetItemFunc(BaseLayer):
    def __init__(self, source_node, module=None, auto_gen=True):
        super(GetItemFunc, self).__init__(source_node, module, auto_gen)

    def add_bottom_top(self, in_names=None, out_names=None):
        pass

    def generate_node(self, name=None, params=None, attr_dict=None):
        if isinstance(self._source_node.args[1], tuple):
            if all(
                isinstance(function, slice) for function in self._source_node.args[1]
            ):
                params_slices = []
                for idx, function in enumerate(self._source_node.args[1]):
                    if (
                        function.start is None
                        and function.stop is None
                        and function.step is None
                    ):
                        continue
                    else:
                        start_ = function.start if function.start is not None else 0
                        end_ = (
                            function.stop if function.stop is not None else sys.maxsize
                        )  # maybe a bug
                        axes_ = idx
                        step_ = function.step if function.step is not None else 1

                        params_slice = [
                            np.array([start_]),
                            np.array([end_]),
                            np.array([axes_]),
                            np.array([step_]),
                        ]
                        params_slices.append(params_slice)
                if len(params_slices) == 1:
                    slice_layer = SliceFunc(self._source_node, auto_gen=False)
                    slice_layer.add_bottom_top()
                    slice_layer.generate_node(params=params_slices[0])
                    self.node_post_process(slice_layer)
                else:
                    slice_layer = SliceFunc(self._source_node, auto_gen=False)
                    slice_layer.add_bottom_top(out_names=[self._name + "_slice_0"])
                    slice_layer.generate_node(
                        name=self._name + "_slice_0", params=params_slices[0]
                    )
                    self.node_post_process(slice_layer)

                    for idx in range(1, len(params_slices) - 1):
                        slice_layer = SliceFunc(self._source_node, auto_gen=False)
                        slice_layer.add_bottom_top(
                            in_names=[self._name + "_slice_" + str(idx - 1)],
                            out_names=[self._name + "_slice_" + str(idx)],
                        )
                        slice_layer.generate_node(
                            name=self._name + "_slice_" + str(idx),
                            params=params_slices[idx],
                        )
                        self.node_post_process(slice_layer)

                    slice_layer = SliceFunc(self._source_node, auto_gen=False)
                    slice_layer.add_bottom_top(
                        in_names=[self._name + "_slice_" + str(len(params_slices) - 2)]
                    )
                    slice_layer.generate_node(
                        name=self._name + "_slice_" + str(len(params_slices) - 1),
                        params=params_slices[len(params_slices) - 1],
                    )
                    self.node_post_process(slice_layer)
            else:
                raise
        else:
            logger.info("getitem_layer: " + self._name + " not supported")
