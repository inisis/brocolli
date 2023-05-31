import re
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


def get_function_name(node_target):
    function_name = re.findall(
        r"(?:function|method) ([a-z|_|0-9]+.*?)", str(node_target)
    )[0]

    return function_name


def create_target(graph_module, node):
    num = 0
    candidate = node.target
    while 1:
        try:
            target_mod = graph_module.graph.owning_module.get_submodule(candidate)
            candidate = f"{node.target}_{num}"
            num += 1
        except Exception as e:
            break

    return candidate


def check_result(actual, desired):
    assert len(actual) == len(desired), "actual: %d vs desired %d" % (
        len(actual),
        len(desired),
    )

    for idx in range(len(actual)):
        np.testing.assert_allclose(
            actual[idx].cpu().detach().numpy(),
            desired[idx].cpu().detach().numpy(),
            rtol=1e-7,
            atol=1e-3,
        )


def _node_dict(graph_module):
    return {n.name: n for n in graph_module.graph.nodes}


def plot_hist(module, float_node, quant_node, quanted_name):
    import matplotlib.pyplot as plt

    float_data = float_node.meta["tensor_meta"]["tensor"]
    quant_data = quant_node.meta["tensor_meta"]["tensor"] * module.output_scale
    float_shape = float_node.meta["tensor_meta"]["shape"]
    quant_shape = quant_node.meta["tensor_meta"]["shape"]
    if hasattr(module, "float_weight"):
        plt.figure(frameon=False, clear=True)
        plt.subplot(2, 2, 1)
        plt.title(
            f"{quanted_name} float weight, shape:{module.float_weight.shape}", wrap=True
        )
        plt.hist(module.float_weight.flatten().numpy(), bins=256)
        plt.subplot(2, 2, 2)
        plt.title(
            f"{quanted_name} quantization weight, shape:{module.weight.shape}",
            wrap=True,
        )
        plt.hist(
            (module.weight.permute(1, 2, 3, 0) * module.wt_scale.flatten(0))
            .flatten()
            .numpy(),
            bins=256,
        )
        plt.subplot(2, 2, 3)
        plt.title(f"{quanted_name} float data, shape:{float_shape}", wrap=True)
        plt.hist(float_data.flatten().numpy(), bins=256)
        plt.subplot(2, 2, 4)
        plt.title(f"{quanted_name} quantization data, shape:{quant_shape}", wrap=True)
        plt.hist(quant_data.flatten().numpy(), bins=256)
        plt.ioff()
        plt.tight_layout(pad=2, w_pad=3, h_pad=3)
        plt.show()
        plt.savefig(f"{quanted_name}.jpg")
        plt.close()
        # for i in range(float_shape[1]):
        #     plt.figure(frameon=False, clear=True)
        #     plt.subplot(1, 2, 1)
        #     plt.hist(float_data[:,i,:,:].flatten().numpy(), bins=256)
        #     plt.subplot(1, 2, 2)
        #     plt.hist(quant_data[:,i,:,:].flatten().numpy(), bins=256)
        #     plt.ioff()
        #     plt.tight_layout(pad=2, w_pad=3, h_pad=3)
        #     plt.show()
        #     plt.savefig(f'{quanted_name}_channel_{i}.jpg')
        #     plt.close()
    else:
        plt.figure(frameon=False, clear=True)
        plt.subplot(1, 2, 1)
        plt.title(f"{quanted_name} float data, shape:{float_shape}", wrap=True)
        plt.hist(float_data.flatten().numpy(), bins=256)
        plt.subplot(1, 2, 2)
        plt.title(f"{quanted_name} quantization data, shape:{quant_shape}", wrap=True)
        plt.hist(quant_data.flatten().numpy(), bins=256)
        plt.ioff()
        plt.tight_layout(pad=2, w_pad=3, h_pad=3)
        plt.show()
        plt.savefig(f"{quanted_name}.jpg")
        plt.close()
        # for i in range(float_shape[1]):
        #     float_data_channel = float_data[:,i,:,:].flatten()
        #     quant_data_channel = quant_data[:,i,:,:].flatten()
        #     cos_sim = F.cosine_similarity(float_data_channel, quant_data_channel, dim=0)
        #     mre = (
        #         torch.abs(quant_data_channel - float_data_channel).sum()
        #         * 100.0
        #         / torch.abs(float_data_channel).sum()
        #     )
        #     plt.figure(frameon=False, clear=True)
        #     plt.subplot(1, 2, 1)
        #     plt.title(f'mre: {mre} cos_sim: {cos_sim}', wrap=True)
        #     plt.hist(float_data_channel.numpy(), bins=256)
        #     plt.subplot(1, 2, 2)
        #     plt.hist(quant_data_channel.numpy(), bins=256)
        #     plt.ioff()
        #     plt.tight_layout(pad=2, w_pad=3, h_pad=3)
        #     plt.show()
        #     plt.savefig(f'{quanted_name}_channel_{i}.jpg')
        #     plt.close()
