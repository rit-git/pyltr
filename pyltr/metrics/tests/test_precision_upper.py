"""

Testing for PrecisionUpper@K metric.

"""

from . import helpers
import itertools
import numpy as np
import pyltr


class TestPrecisionUpper(helpers.TestMetric):
    def get_metric(self):
        return pyltr.metrics.PrecisionUpper(k=4)

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
        yield [1, None, 1, None, 1], 4.0 / 4
        yield [1, None, 1, 0, 0], 3.0 / 4
        yield [None, None, None, None, 0], 4.0 / 4
        yield [None, None, None, None, None], 4.0 / 4

    def get_queries(self):
        for i in range(0, 11):
            for tup in itertools.product(*([(0, 1)] * i)):
                yield np.array(tup)
