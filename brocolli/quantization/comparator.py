import numbers

import torch
from torch.fx import Interpreter
from torch.fx.node import map_aggregate


class FXComparator(Interpreter):
    def __init__(self, module, dynamic_batch=False):
        super(FXComparator, self).__init__(module)
        self.dynamic_batch = dynamic_batch

    def run_node(self, n):
        result = super().run_node(n)
        if isinstance(result, numbers.Number):
            result = torch.tensor(result)

        found_tensor = False

        def extract_tensor_metadata(result: torch.Tensor):
            meta_info = {}
            shape = list(result.shape)
            if self.dynamic_batch:
                shape[0] = -1

            meta_info["tensor"] = result
            meta_info["shape"] = torch.Size(shape)
            meta_info["dtype"] = result.dtype

            return meta_info

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta["tensor_meta"] = meta

        n.meta["type"] = type(result)
        return result

    def __call__(self, x):
        return self.run(x)
