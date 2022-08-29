import torch


class BaseOperator:
    def clamp(self, x):
        if self.output_min_value < 0:
            threshold_max = 2 ** (self.qbit - 1) - 1
            threshold_min = -(2 ** (self.qbit - 1))
        else:
            threshold_max = 2 ** (self.qbit) - 1
            threshold_min = 0

        x = torch.clamp(x, threshold_min, threshold_max)
        x = x.to(torch.int64)

        return x
