"""

Testing for Precision@K metric.

"""

from . import helpers
import itertools
import numpy as np
import pyltr


class TestPrecision(helpers.TestMetric):
    def get_metric(self):
        return pyltr.metrics.Precision(k=4)

    def get_queries_with_values(self):
        yield [], 0.0
        yield [0], 0.0
        yield [1], 1.0 / 4
        yield [1, 0], 1.0 / 4
        yield [0, 1], 1.0 / 4
        yield [1, 0, 1, 0], 2.0 / 4
        yield [0, 1, 1, 1], 3.0 / 4
        yield [1, 0, 1, 0, 1], 2.0 / 4
        yield [1, 0, 1, 0, 0], 2.0 / 4

    def get_queries(self):
        for i in range(0, 7):
            for tup in itertools.product(*([(0, 1)] * i)):
                yield np.array(tup)
