import time
import numpy as np
import pandas as pd
from tabulate import tabulate
from loguru import logger

from torch.fx.interpreter import Interpreter


class ProfileStats(object):
    def __init__(self):
        super(ProfileStats, self).__init__()
        self.total_time = 0
        self.runtime_info = {}

    def record(self, node, sec):
        """Record timings of a single call"""
        self.total_time += sec
        self.runtime_info.setdefault(node, [])
        self.runtime_info[node].append(sec)

    def summary(self, save_to_disk=False):
        node_summaries = []
        for node, runtime in self.runtime_info.items():
            mean_runtime = np.mean(runtime)
            percent_runtime = np.sum(runtime) / self.total_time * 100
            if node.op == "call_function":
                node_summaries.append(
                    [node.op, node.name, mean_runtime, percent_runtime]
                )
            else:
                node_summaries.append(
                    [node.op, node.target, mean_runtime, percent_runtime]
                )

        node_summaries.sort(key=lambda s: s[2], reverse=True)
        headers = [
            "\nopcode",
            "\nname",
            "\nruntime (s)",
            "\npercentage (%)",
        ]
        logger.debug(tabulate(node_summaries, headers=headers, numalign="left"))
        if save_to_disk:
            with open("profile_stats.txt", "w") as f:
                f.write(tabulate(node_summaries, headers=headers))


class FXProfiler(Interpreter):
    def __init__(self, module):
        super(FXProfiler, self).__init__(module)
        self.profiler = ProfileStats()

    def run_node(self, n):
        """Timing wrapper around executing an FX Node"""
        start = time.perf_counter()
        result = super().run_node(n)
        sec = time.perf_counter() - start
        self.profiler.record(n, sec)

        return result
