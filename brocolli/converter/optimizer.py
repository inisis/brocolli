from loguru import logger
import contextlib
import onnx_graphsurgeon as gs
from collections import OrderedDict, Counter

DEFAULT_FUSION_PATTERNS = OrderedDict()


def register_fusion_pattern(layer_type):
    def insert(fn):
        if layer_type in DEFAULT_FUSION_PATTERNS.keys():
            raise
        DEFAULT_FUSION_PATTERNS[layer_type] = fn
        return fn

    return insert


def get_default_fusion_patterns():
    return DEFAULT_FUSION_PATTERNS


def graph_constant_fold_inplace(graph):
    for node in graph.nodes:
        if node.op == "Identity" or node.op == "Dropout":
            inp_node = node.i()
            inp_node.outputs = node.outputs
            node.outputs.clear()


@register_fusion_pattern("GELU")
def find_gelu_nodes(node):
    # fmt: off
    '''
             x
         /      \
         |     Div
         |      |
         |     Erf
         |      |
         |     Add
         \      /
            Mul
             |
            Mul
    '''
    # fmt: on
    match = {}
    if node.op == "Mul":
        if (
            node.i(0).op == "Mul"
            and node.i(0).i(1).op == "Add"
            and node.i(0).i(1).i(0).op == "Erf"
            and node.i(0).i(1).i(0).i(0).op == "Div"
        ):
            input_variable = node.i(0).i(1).i(0).i(0).inputs[0]
            mul_node = node.i(0)
            div_node = node.i(0).i(1).i(0).i(0)

            input_variable.outputs.remove(mul_node)
            input_variable.outputs.remove(div_node)

            output_variable = node.outputs[0]
            output_variable.inputs.clear()
            match.update(
                {
                    "inputs": [input_variable],
                    "outputs": [output_variable],
                }
            )

    return match


@register_fusion_pattern("Swish")
def find_swish_nodes(node):
    # fmt: off
    '''
             x
         /      \
         |    Sigmoid
         \      /
            Mul
    '''
    # fmt: on
    match = {}
    if node.op == "Mul":
        if node.i(1).op == "Sigmoid":
            input_variable = node.i(1).inputs[0]
            mul_node = node.i(1)
            sigmoid_node = node

            input_variable.outputs.remove(mul_node)
            input_variable.outputs.remove(sigmoid_node)

            output_variable = node.outputs[0]
            output_variable.inputs.clear()
            match.update(
                {
                    "inputs": [input_variable],
                    "outputs": [output_variable],
                }
            )

    return match


@register_fusion_pattern("LayerNormalization")
def find_layernorm_nodes(node):
    # fmt: off
    '''
             x
         /      \
         |  ReduceMean
         \      /
            Sub
         /      \
         |     Pow
         |      |
         |  ReduceMean
         |      |
         |     Add
         |      |
         |     Sqrt
         \      /
            Div
             |
            Mul
             |
            Add
    '''
    # fmt: on
    match = {}
    if node.op == "Add":
        if (
            node.i(0).op == "Mul"
            and node.i(0).i(0).op == "Div"
            and node.i(0).i(0).i(0).op == "Sub"
            and node.i(0).i(0).i(1).op == "Sqrt"
            and node.i(0).i(0).i(1).i(0).op == "Add"
            and node.i(0).i(0).i(1).i(0).i(0).op == "ReduceMean"
            and node.i(0).i(0).i(1).i(0).i(0).i(0).op == "Pow"
            and node.i(0).i(0).i(1).i(0).i(0).i(0).i(0).op == "Sub"
            and node.i(0).i(0).i(1).i(0).i(0).i(0).i(0).i(1).op == "ReduceMean"
        ):
            input_variable = node.i(0).i(0).i(1).i(0).i(0).i(0).i(0).i(1).inputs[0]
            sub_node = node.i(0).i(0).i(1).i(0).i(0).i(0).i(0).i(1)
            reducemean_node = node.i(0).i(0).i(1).i(0).i(0).i(0).i(0)

            input_variable.outputs.remove(sub_node)
            input_variable.outputs.remove(reducemean_node)

            weight_variable = node.i(0).inputs[1]
            bias_variable = node.inputs[1]

            output_variable = node.outputs[0]
            output_variable.inputs.clear()
            match.update(
                {
                    "inputs": [
                        input_variable,
                        weight_variable,
                        bias_variable,
                    ],
                    "outputs": [output_variable],
                    "attrs": {
                        "attrs": str(
                            {
                                "axis": node.i(0).i(0).i(1).i(0).i(0).attrs["axes"][0],
                                "eps": float(node.i(0).i(0).i(1).i(0).inputs[1].values),
                            }
                        )
                    },
                }
            )
    return match


@gs.Graph.register()
def replace_custom_layer(
    self, op, inputs, outputs, name, attrs=None, domain="ai.onnx.contrib"
):
    return self.layer(
        op=op,
        inputs=inputs,
        outputs=outputs,
        name=name,
        attrs=attrs,
        domain=domain,
    )


def find_matches(graph, fusion_patterns):
    match_map = {}
    counter = Counter()
    for node in reversed(graph.nodes):
        if node.name not in match_map:
            for layer_type, func in fusion_patterns.items():
                with contextlib.suppress(IndexError):
                    match = func(node)
                    if match:
                        logger.debug("matched patter {}", layer_type)
                        match.update({"op": layer_type})
                        match.update(
                            {
                                "name": "{}_{}".format(
                                    layer_type.lower(), counter[layer_type]
                                )
                            }
                        )
                        counter.update([layer_type])
                        # the first pattern matches will take precedence
                        if node.name not in match_map:
                            match_map[node.name] = match

    return match_map


def optimize_model(model):
    graph = gs.import_onnx(model)
    graph.fold_constants().cleanup()
    fusion_patterns = get_default_fusion_patterns()
    fusion_pairs = find_matches(graph, fusion_patterns)
    for _, match in fusion_pairs.items():
        graph.replace_custom_layer(**match)
    graph_constant_fold_inplace(graph)
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    model = gs.export_onnx(graph)

    return model
