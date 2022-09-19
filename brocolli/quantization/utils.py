import numpy as np


def activation_pre_hook(self, input):
    if hasattr(self, "activation_pre_process"):
        self.activation_pre_process(input[0])


def activation_post_hook(self, input, output):
    if hasattr(self, "activation_post_process"):
        self.activation_post_process(output)


def _parent_name(target):
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def replace_node_module(node, modules, new_module):
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, new_module)


def check_result(actual, desired):
    assert len(actual) == len(desired), "actual: %d vs desired %d" % (
        len(actual),
        len(desired),
    )

    for idx in range(len(actual)):
        np.testing.assert_allclose(
            actual[idx].detach().numpy(),
            desired[idx].detach().numpy(),
            rtol=1e-7,
            atol=1e-3,
        )
