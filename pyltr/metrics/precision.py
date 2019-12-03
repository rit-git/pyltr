"""

Precision@K

TODO: better docs

"""

import numpy as np
from . import Metric
from sklearn.externals.six.moves import range


class Precision(Metric):
    def __init__(self, k=10, cutoff=0.5):
        """
        target >= cutoff means relevant, not relevant otherwise
        """
        if k < 0:
            raise ValueError('k must be positive')
        super(Precision, self).__init__()
        self.k = k
        self.cutoff = cutoff

    def evaluate(self, qid, targets):
        n_targets = len(targets)
        num_rel = 0.
        for i in range(n_targets):
            if i >= self.k:
                break
            if targets[i] >= self.cutoff:
                num_rel += 1
        return (num_rel / self.k)

    def max_k(self):
        return self.k
