import contextlib
import onnx_graphsurgeon as gs


def graph_constant_fold_inplace(graph):
    for node in graph.nodes:
        if node.op == "Identity" or node.op == "Dropout":
            inp_node = node.i()
            inp_node.outputs = node.outputs
            node.outputs.clear()


def find_gelu_nodes(graph):
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
    out_nodes = []
    for node in graph.nodes:
        with contextlib.suppress(IndexError):
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
                    out_nodes += [
                        {
                            "inps": [input_variable],
                            "outs": [output_variable],
                        }
                    ]

    return out_nodes


def find_swish_nodes(graph):
    # fmt: off
    '''
             x
         /      \
         |    Sigmoid
         \      /
            Mul
    '''
    # fmt: on
    out_nodes = []
    for node in graph.nodes:
        with contextlib.suppress(IndexError):
            if node.op == "Mul":
                if node.i(1).op == "Sigmoid":
                    input_variable = node.i(1).inputs[0]
                    mul_node = node.i(1)
                    sigmoid_node = node

                    input_variable.outputs.remove(mul_node)
                    input_variable.outputs.remove(sigmoid_node)

                    output_variable = node.outputs[0]
                    output_variable.inputs.clear()
                    out_nodes += [
                        {
                            "inps": [input_variable],
                            "outs": [output_variable],
                        }
                    ]

    return out_nodes


def find_layernorm_nodes(graph):
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
    out_nodes = []
    for node in graph.nodes:
        with contextlib.suppress(IndexError):
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
                    input_variable = (
                        node.i(0).i(0).i(1).i(0).i(0).i(0).i(0).i(1).inputs[0]
                    )
                    sub_node = node.i(0).i(0).i(1).i(0).i(0).i(0).i(0).i(1)
                    reducemean_node = node.i(0).i(0).i(1).i(0).i(0).i(0).i(0)

                    input_variable.outputs.remove(sub_node)
                    input_variable.outputs.remove(reducemean_node)

                    weight_variable = node.i(0).inputs[1]
                    bias_variable = node.inputs[1]

                    output_variable = node.outputs[0]
                    output_variable.inputs.clear()
                    out_nodes += [
                        {
                            "inps": [
                                input_variable,
                                weight_variable,
                                bias_variable,
                            ],
                            "outs": [output_variable],
                            "attrs": {
                                "attrs": str(
                                    {
                                        "axis": node.i(0)
                                        .i(0)
                                        .i(1)
                                        .i(0)
                                        .i(0)
                                        .attrs["axes"][0],
                                        "eps": float(
                                            node.i(0).i(0).i(1).i(0).inputs[1].values
                                        ),
                                    }
                                )
                            },
                        }
                    ]
    return out_nodes


@gs.Graph.register()
def replace_gelu(self, inputs, outputs, name):
    return self.layer(
        op="GELU", inputs=inputs, outputs=outputs, name=name, domain="ai.onnx.contrib"
    )


@gs.Graph.register()
def replace_swish(self, inputs, outputs, name):
    return self.layer(
        op="Swish", inputs=inputs, outputs=outputs, name=name, domain="ai.onnx.contrib"
    )


@gs.Graph.register()
def replace_layernorm(self, inputs, outputs, attrs, name):
    return self.layer(
        op="LayerNormalization",
        inputs=inputs,
        outputs=outputs,
        name=name,
        attrs=attrs,
        domain="ai.onnx.contrib",
    )


def optimize_model(model):
    graph = gs.import_onnx(model)
    graph.fold_constants().cleanup()
    gelu_nodes = find_gelu_nodes(graph)
    swish_node = find_swish_nodes(graph)
    layernorm_node = find_layernorm_nodes(graph)

    for i, itn in enumerate(gelu_nodes):
        inputs = itn["inps"]
        outputs = itn["outs"]
        name = "gelu_{}".format(i)
        graph.replace_gelu(inputs, outputs, name)

    for i, itn in enumerate(swish_node):
        inputs = itn["inps"]
        outputs = itn["outs"]
        name = "swish_{}".format(i)
        graph.replace_swish(inputs, outputs, name)

    for i, itn in enumerate(layernorm_node):
        inputs = itn["inps"]
        outputs = itn["outs"]
        attrs = itn["attrs"]
        name = "layernorm_{}".format(i)
        graph.replace_layernorm(inputs, outputs, attrs, name)

    graph_constant_fold_inplace(graph)
    graph.cleanup(remove_unused_graph_inputs=True).toposort()
    model = gs.export_onnx(graph)

    return model
