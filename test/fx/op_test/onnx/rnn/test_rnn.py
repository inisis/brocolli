import torch
import torch.nn as nn
import pytest
import warnings

from bin.fx.utils import OnnxBaseTester as Tester

def test_RNN_basic(shape = [5, 3, 10], opset_version=13):
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(RNN, self).__init__()
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        def forward(self,x):
            return self.rnn(x)[0]

    model = RNN(10, 20, 2)
    Tester("RNN_basic", model, shape, opset_version)

def test_RNN_2inputs(shape = ([5, 3, 10], [2, 3, 20]), opset_version=13):
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(RNN, self).__init__()
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        def forward(self,x, y):
            return self.rnn(x, y)[0]

    model = RNN(10, 20, 2)
    Tester("RNN_2inputs", model, shape, opset_version)

def test_RNN_2inputs_2outputs(shape = ([5, 3, 10], [2, 3, 20]), opset_version=13):
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(RNN, self).__init__()
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        def forward(self,x, y):
            return self.rnn(x, y)

    model = RNN(10, 20, 2)
    Tester("RNN_2inputs_2outputs", model, shape, opset_version)

def test_RNN_2inputs_2outputs_batch_first(shape = ([5, 3, 10], [1, 5, 20]), opset_version=13):
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(RNN, self).__init__()
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        def forward(self,x, y):
            return self.rnn(x, y)

    model = RNN(10, 20, 1)
    Tester("RNN_2inputs_2outputs_batch_first", model, shape, opset_version)

def test_LSTM_basic(shape = [5, 3, 10], opset_version=13):
    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(10, 20, 1)

        def forward(self,x):
            return self.lstm(x)[0]

    model = LSTM()
    Tester("LSTM_basic", model, shape, opset_version)

def test_LSTM_bidirectional(shape = [5, 3, 10], opset_version=13):
    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(10, 20, 1, bidirectional=True)

        def forward(self,x):
            return self.lstm(x)[0]
    model = LSTM()
    Tester("LSTM_bidirectional", model, shape, opset_version)

def test_LSTM_bias(shape = [5, 3, 10], opset_version=13):
    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(10, 20, 1, bidirectional=True, bias=True)

        def forward(self,x):
            return self.lstm(x)[0]  
    model = LSTM()
    Tester("LSTM_bias", model, shape, opset_version)

def test_LSTM_multilayer(shape = [5, 3, 10], opset_version=13):
    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(10, 20, 2)

        def forward(self,x):
            return self.lstm(x)[0]

    model = LSTM()
    Tester("LSTM_multilayer", model, shape, opset_version)

def test_LSTM_2inputs(shape = ([5, 3, 10], ([2, 3, 20], [2, 3, 20])), opset_version=13):
    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(10, 20, 2)

        def forward(self,x, y):
            return self.lstm(x, y)[0]

    model = LSTM()
    Tester("LSTM_2inputs", model, shape, opset_version)

def test_LSTM_2inputs_2outputs(shape = ([5, 3, 10], ([2, 3, 20], [2, 3, 20])), opset_version=13):
    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(10, 20, 2)

        def forward(self,x, y):
            return self.lstm(x, y)

    model = LSTM()
    Tester("LSTM_2inputs_2outputs", model, shape, opset_version)

def test_LSTM_2inputs_2outputs_batch_first(shape = ([5, 3, 10], ([1, 5, 20], [1, 5, 20])), opset_version=13):
    class LSTM(nn.Module):
        def __init__(self):
            super(LSTM, self).__init__()
            self.lstm = nn.LSTM(10, 20, 1, batch_first=True)

        def forward(self,x, y):
            return self.lstm(x, y)

    model = LSTM()
    Tester("LSTM_2inputs_2outputs_batch_first", model, shape, opset_version)

def test_GRU_basic(shape = [5, 3, 10], opset_version=13):
    class GRU(nn.Module):
        def __init__(self):
            super(GRU, self).__init__()
            self.gru = nn.GRU(10, 20, 2, bidirectional=True, bias=True)

        def forward(self,x):
            return self.gru(x)[0]  
    model = GRU()
    Tester("GRU_basic", model, shape, opset_version)

def test_GRU_2inputs(shape = ([5, 3, 10], [2, 3, 20]), opset_version=13):
    class GRU(nn.Module):
        def __init__(self):
            super(GRU, self).__init__()
            self.gru = nn.GRU(10, 20, 2)

        def forward(self,x, y):
            return self.gru(x, y)[0]

    model = GRU()
    Tester("GRU_2inputs", model, shape, opset_version)

def test_GRU_2inputs_2outputs(shape = ([5, 3, 10], [2, 3, 20]), opset_version=13):
    class GRU(nn.Module):
        def __init__(self):
            super(GRU, self).__init__()
            self.gru = nn.GRU(10, 20, 2)

        def forward(self,x, y):
            return self.gru(x, y)

    model = GRU()
    Tester("GRU_2inputs_2outputs", model, shape, opset_version)

def test_GRU_2inputs_2outputs_batch_first(shape = ([5, 3, 10], [2, 5, 20]), opset_version=13):
    class GRU(nn.Module):
        def __init__(self):
            super(GRU, self).__init__()
            self.gru = nn.GRU(10, 20, 2, batch_first=True)

        def forward(self,x, y):
            return self.gru(x, y)

    model = GRU()
    Tester("GRU_2inputs_2outputs_batch_first", model, shape, opset_version)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pytest.main(['-p', 'no:warnings', '-v', 'test/op_test/onnx/linear/test_rnn.py'])
