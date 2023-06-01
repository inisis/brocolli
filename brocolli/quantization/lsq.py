import torch
import torch.nn as nn

from loguru import logger


class LSQER(nn.Module):
    def __init__(self, float_module, lsq_module):
        super(LSQER, self).__init__()
        self.float_module = float_module
        self.lsq_module = lsq_module
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.lsq_module.parameters()),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0001,
        )

    def _float_forward(self, x):
        with torch.no_grad():
            output_float = self.float_module(x)

        return output_float

    def _forward(self, x):
        output_float = self._float_forward(x)
        with torch.enable_grad():
            output_lsq = self.lsq_module(x)
            loss = self.criterion(output_float, output_lsq)
            logger.debug("lsq loss: ", loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return output_float, output_lsq

    def forward(self, x):
        output_float, output_lsq = self._forward(x)

        return output_lsq

    def train(self):
        self.float_module.eval()
        self.lsq_module.train()

    def eval(self):
        self.float_module.eval()
        self.lsq_module.train()
