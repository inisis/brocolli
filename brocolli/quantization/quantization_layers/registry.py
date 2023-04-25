from collections import OrderedDict

DEFAULT_QUANT_OPS = OrderedDict()


def register_quant_op(op):
    def insert(fn):
        if op in DEFAULT_QUANT_OPS.keys():
            raise
        DEFAULT_QUANT_OPS[op] = fn
        return fn

    return insert


def get_default_quant_ops():
    return DEFAULT_QUANT_OPS
