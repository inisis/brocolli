import torch
import pytest
import warnings

from brocolli.testing.common_utils import OnnxBaseTester as Tester


class TestRNNClass:
    @pytest.mark.parametrize("multi_inputs", (True, False))
    @pytest.mark.parametrize("multi_outputs", (True, False))
    @pytest.mark.parametrize("batch_first", (True, False))
    def test_RNN(
        self,
        request,
        multi_inputs,
        multi_outputs,
        batch_first,
        shape=((2, 2, 10), (2, 2, 20)),
    ):
        class RNN(torch.nn.Module):
            def __init__(
                self, input_size, hidden_size, num_layers, batch_first, multi_outputs
            ):
                super(RNN, self).__init__()
                self.multi_outputs = multi_outputs
                self.rnn = torch.nn.RNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=batch_first,
                )

            def forward(self, x):
                if self.multi_outputs:
                    return self.rnn(x)
                else:
                    return self.rnn(x)[0]

        class RNN_(torch.nn.Module):
            def __init__(
                self, input_size, hidden_size, num_layers, batch_first, multi_outputs
            ):
                super(RNN_, self).__init__()
                self.multi_outputs = multi_outputs
                self.rnn = torch.nn.RNN(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=batch_first,
                )

            def forward(self, x, y):
                if self.multi_outputs:
                    return self.rnn(x, y)
                else:
                    return self.rnn(x, y)[0]

        if multi_inputs:
            x = torch.rand(shape[0])
            y = torch.rand(shape[1])
            model = RNN_(10, 20, 2, batch_first, multi_outputs)
            Tester(request.node.name, model, (x, y))
        else:
            x = torch.rand(shape[0])
            model = RNN(10, 20, 2, batch_first, multi_outputs)
            Tester(request.node.name, model, x)

    @pytest.mark.parametrize("multi_inputs", (True, False))
    @pytest.mark.parametrize("multi_outputs", (True, False))
    @pytest.mark.parametrize("bidirectional", (True, False))
    def test_GRU(
        self,
        request,
        multi_inputs,
        multi_outputs,
        bidirectional,
        shape=((2, 2, 10), (2, 2, 20)),
    ):
        class GRU(torch.nn.Module):
            def __init__(self, bidirectional, multi_outputs):
                super(GRU, self).__init__()
                self.multi_outputs = multi_outputs
                self.gru = torch.nn.GRU(
                    10, 20, 2, bidirectional=bidirectional, bias=True
                )

            def forward(self, x):
                if self.multi_outputs:
                    return self.gru(x)
                else:
                    return self.gru(x)[0]

        class GRU_(torch.nn.Module):
            def __init__(self, bidirectional, multi_outputs):
                super(GRU_, self).__init__()
                self.multi_outputs = multi_outputs
                self.gru = torch.nn.GRU(
                    10, 20, 2, bidirectional=bidirectional, bias=True
                )

            def forward(self, x, y):
                if self.multi_outputs:
                    return self.gru(x, y)
                else:
                    return self.gru(x, y)[0]

        if multi_inputs:
            x = torch.rand(shape[0])
            if bidirectional:
                y = torch.rand((shape[1][0] * 2, shape[1][1], shape[1][2]))
            else:
                y = torch.rand(shape[1])
            model = GRU_(bidirectional, multi_outputs)
            Tester(request.node.name, model, (x, y))
        else:
            x = torch.rand(shape[0])
            model = GRU(bidirectional, multi_outputs)
            Tester(request.node.name, model, x)

    @pytest.mark.parametrize("multi_inputs", (True, False))
    @pytest.mark.parametrize("multi_outputs", (True, False))
    @pytest.mark.parametrize("bidirectional", (True, False))
    @pytest.mark.parametrize("num_layers", (1, 2))
    @pytest.mark.parametrize("bias", (True, False))
    def test_LSTM(
        self,
        request,
        multi_inputs,
        multi_outputs,
        bidirectional,
        num_layers,
        bias,
        shape=((2, 2, 10), ((2, 2, 20), (2, 2, 20))),
    ):
        class LSTM(torch.nn.Module):
            def __init__(self, num_layers, bidirectional, bias, multi_outputs):
                super(LSTM, self).__init__()
                self.multi_outputs = multi_outputs
                self.lstm = torch.nn.LSTM(
                    10, 20, num_layers, bidirectional=bidirectional, bias=bias
                )

            def forward(self, x):
                if self.multi_outputs:
                    return self.lstm(x)
                else:
                    return self.lstm(x)[0]

        class LSTM_(torch.nn.Module):
            def __init__(self, num_layers, bidirectional, bias, multi_outputs):
                super(LSTM_, self).__init__()
                self.multi_outputs = multi_outputs
                self.lstm = torch.nn.LSTM(
                    10, 20, num_layers, bidirectional=bidirectional, bias=bias
                )

            def forward(self, x, y):
                if self.multi_outputs:
                    return self.lstm(x, y)
                else:
                    return self.lstm(x, y)[0]

        if multi_inputs:
            x = torch.rand(shape[0])
            if bidirectional:
                y = (
                    torch.rand((num_layers * 2, shape[1][0][1], shape[1][0][2])),
                    torch.rand((num_layers * 2, shape[1][1][1], shape[1][1][2])),
                )
            else:
                y = (
                    torch.rand((num_layers, shape[1][0][1], shape[1][0][2])),
                    torch.rand((num_layers, shape[1][1][1], shape[1][1][2])),
                )
            model = LSTM_(num_layers, bidirectional, bias, multi_outputs)
            Tester(request.node.name, model, (x, y))
        else:
            x = torch.rand(shape[0])
            model = LSTM(num_layers, bidirectional, bias, multi_outputs)
            Tester(request.node.name, model, x)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    pytest.main(
        ["-p", "no:warnings", "-v", "test/converter/op_test/onnx/rnn/test_rnn.py"]
    )
