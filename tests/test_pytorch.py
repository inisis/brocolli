from __future__ import absolute_import
from __future__ import print_function

import os
import sys

from converter.pytorch.pytorch_parser import PytorchParser

model_file = "model/pnet_epoch_model_10.pkl"

parser = PytorchParser("model/pnet_epoch_model_10.pkl", [3, 12, 12])

parser.run(model_file)
